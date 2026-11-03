"""gpuq — a numbered GPU queue for a shared machine, with a terminal view.

GPU users:  ID=$(gpuq join --owner NAME --job TEXT --why TEXT --models a,b --est MIN --max MIN)
            gpuq start ID     (exit 3 = not your turn yet; retry in a minute)
            gpuq beat ID [--progress 41/80]   at least every 15 min (exit 4 = you were purged: stop;
                                              exit 5 = yield: stop your GPU work, go back to polling start)
            gpuq done ID
            or all of that in one:  gpuq run ID [--on-stop CMD] [--progress-file F] -- CMD ...
            Yield: moving the running job down (a human, or `gpuq yield ID`) makes its next beat exit 5.
            Nobody may start until it acknowledges — by calling start (exit 3, it is waiting again)
            or done. When it is #1 again, its own start loop resumes it: no "I'm done" message needed.
Humans:     gpuq                          show the line once
            gpuq tui --hosts a,b          interactive: ↑↓ select, k/j (or Shift-↑↓) move up/down, d remove
            gpuq watch --hosts a,b        plain live view of several machines
            gpuq mv 3 1  ·  gpuq rm 4     reorder / remove (by position or id); moving the running
                                          job, or anything above it, asks it to yield
            gpuq yield ID [N]             ask the running job to step back to #N (default 2)
            gpuq pause  ·  gpuq resume    hold the line (nobody may start) / let it go
            gpuq evac [CARD] [--to C]     move a card's resident services off it (free the whole card)
            gpuq restore [CARD]           put the residents back ([--dry-run] on either to preview)
            gpuq --host a mv 3 1          do it on another machine over SSH
Timer:      gpuq sweep    every minute: purge a holder silent 15 min or past its max;
                          skip a #1 that hasn't started within 15 min (twice → removed)
Several cards: every card has its own line, named by its model number (e.g. 3090, 3060).
            `gpuq join --gpu 3060 ...` picks a line; without --gpu: the owner's default from
            the config file, else the host default (else the card with most memory).
            Ids work across lines; positions (`mv 3 1`, `rm 4`) are in the default line
            unless you add --gpu. `gpuq run` hands the job its card in CUDA_VISIBLE_DEVICES
            (by UUID — the index order differs between nvidia-smi and CUDA); `start` prints it.

Config (optional JSON, ~/.config/gpuq/config.json or $GPUQ_CONFIG): see examples/config.json.
"""
import argparse
import fcntl
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .core import *  # noqa: F401,F403  (pure queue operations, re-exported for callers)
from .core import QueueError, fmt_dur, yielding, running, normalize, find, remove

HOST = socket.gethostname()
STATE = Path(os.environ.get("GPUQ_STATE") or Path.home() / ".local" / "state" / "gpuq")
SYSFS = Path(os.environ.get("GPUQ_SYSFS") or "/sys")     # roots are overridable so tests can
PROC = Path(os.environ.get("GPUQ_PROC") or "/proc")      # use fixture trees
CONFIG = Path(os.environ.get("GPUQ_CONFIG") or Path.home() / ".config" / "gpuq" / "config.json")
# Optional lock file for tools that predate gpuq: written while a job runs, removed when it ends
# (only if gpuq wrote it). Set `compat_lock` in the config, or $GPUQ_COMPAT_LOCK. "" disables.
COMPAT_LOCK = os.environ.get("GPUQ_COMPAT_LOCK")     # None = ask the config
# Per-host settings (all optional), see examples/config.json:
#   default, owners, residents, units, hosts, compat_lock, remote_command
# Residents are always-on services that live on a card outside the queue: never "unqueued use",
# flagged if on the wrong card.


# ---- formatting ----

def fmt_clock(t, now):
    lt, ln = time.localtime(t), time.localtime(now)
    same_day = lt[:3] == ln[:3]
    return time.strftime("%H:%M" if same_day else "%a %H:%M", lt)


def _trim(s, n):
    """Shorten to n cells for the one-line views, marking the cut with an ellipsis."""
    return s if len(s) <= n else s[:n - 1] + "…"


def snapshot(q, now, hist, loaded, host=HOST, paused=None, lines=None, cards=None, card_use=None,
             default=None, evac=None):
    """`entries` is always the default card's line (what older readers know); `lines` has every
    card's, default first; `card_use` = {card: {"jobs", "residents", "misplaced"}}.
    `default` names the default card — the one the local host marks as "you" in a multi-card view.
    `evac` is the current evac record ({card, to, ...}) or None."""
    return {"host": host, "now": now, "entries": q, "history": hist, "loaded": loaded,
            "foreign_lock": foreign_compat(), "paused": paused, "lines": lines or {},
            "cards": cards or {}, "card_use": card_use or {}, "default": default, "evac": evac}


def per_card(snap):
    """One view per card's line, each a snapshot of that card alone. A one-line snapshot (a one-card host,
    an older remote, the tests) is its own only view."""
    lines = snap.get("lines") or {}
    if len(lines) < 2:
        return [snap]
    out = []
    for name, q in lines.items():
        use = (snap.get("card_use") or {}).get(name)
        out.append(dict(snap, entries=q, card=name, loaded=None if use is None else use["jobs"],
                        residents=(use or {}).get("residents", []),
                        misplaced=(use or {}).get("misplaced", []),
                        busy=(use or {}).get("busy")))
    return out


def header(snap, now):
    q = snap["entries"]
    where = snap["host"]
    if snap.get("card"):
        c = (snap.get("cards") or {}).get(snap["card"]) or {}
        gib = c.get("mib")
        where += " %s%s" % (snap["card"], " (%d GB%s)" % (round(gib / 1024), " shared" if c.get("integrated") else "")
                            if gib else "")
    text = "%s · %s · %s" % (where, fmt_clock(now, now),
                             "%d in line" % len(q) if q else "queue empty")
    if snap.get("paused"):
        text += " · ⏸ PAUSED since %s — nobody may start" % fmt_clock(snap["paused"]["since"], now)
    ev = snap.get("evac")
    if ev and ev.get("card") == snap.get("card"):
        text += " · ⏏ EVACUATED → %s" % ev.get("to")
    return text


def entry_timings(q, now):
    """[(entry, timing text)] — elapsed/ETA for the runner, a start estimate for the rest.
    The start estimate chains through the line and stops at the first missing estimate."""
    out, cursor = [], now
    for e in q:
        if e["state"] == "running":
            elapsed = now - e["started"]
            if e.get("est_min"):
                cursor = now + max(e["est_min"] * 60 - elapsed, 0)
                t = "%s of ~%s, ends ~%s" % (fmt_dur(elapsed), fmt_dur(e["est_min"] * 60),
                                             fmt_clock(cursor, now))
            else:
                cursor, t = None, "%s so far, no estimate" % fmt_dur(elapsed)
        elif e["state"] == "yielding":
            t = "yielding, asked %s ago — stopping, then waits here" % fmt_dur(now - e["yield_since"])
        else:
            est = "~%s" % fmt_dur(e["est_min"] * 60) if e.get("est_min") else "no estimate"
            if cursor is not None:
                t = "%s, starts ~%s" % (est, fmt_clock(cursor, now))
                cursor = cursor + e["est_min"] * 60 if e.get("est_min") else None
            else:
                t = "%s, starts: unknown" % est
            if e.get("skips"):
                t += " (skipped once)"
        out.append((e, t))
    return out


PALETTE = [36, 35, 33, 32, 34, 31]  # per-GPU colours: cyan, magenta, yellow, green, blue, red


def _sgr(s, on, *codes):
    return "\033[%sm%s\033[0m" % (";".join(str(c) for c in codes), s) if on and codes else s


def _dimc(s, on):
    return _sgr(s, on, 2)


def _termwidth():
    return max(40, min(shutil.get_terminal_size(fallback=(80, 24)).columns, 100))


def _is_current(view):
    """The card the local host counts as 'you': its default card (or its only card)."""
    if view.get("host") != HOST:
        return False
    card = view.get("card")
    return True if card is None else card == view.get("default")


def _body_lines(view, color):
    """Every line of a card's block below its header: the queue entries, then warnings."""
    bold = (lambda s: "\033[1;32m%s\033[0m" % s) if color else (lambda s: s)
    dim = (lambda s: "\033[2m%s\033[0m" % s) if color else (lambda s: s)
    now, q = view["now"], view["entries"]
    out = []
    for i, (e, timing) in enumerate(entry_timings(q, now), 1):
        models = ",".join(e["models"]) or "-"
        if e["state"] == "running":
            beat_age = "beat %s ago" % fmt_dur(now - (e["heartbeat"] or e["started"]))
            line = "%2d ▶ RUNNING %s%-12s %-28s %s · %s · %s" % (
                i, "★" if e.get("important") else " ", e["owner"][:12], _trim(e["job"], 28), models,
                timing, beat_age)
            if e.get("progress"):
                line += " · " + e["progress"]
            out.append(bold(line))
        elif e["state"] == "yielding":
            out.append("%2d ⏏ YIELDING %-12s %-28s %s · %s · beat %s ago" % (
                i, e["owner"][:12], _trim(e["job"], 28), models, timing,
                fmt_dur(now - (e["heartbeat"] or e["started"]))))
        else:
            out.append("%2d   waiting %s%-12s %-28s %s · %s" % (
                i, "★" if e.get("important") else " ", e["owner"][:12], _trim(e["job"], 28), models, timing))
        if e.get("why"):
            out.append(dim("       why: %s" % e["why"]))
    out.extend(warnings(view))
    return out


def _history_block(snap, color, heading=True):
    hist = snap.get("history", [])[-5:]
    if not hist:
        return []
    out = [_dimc("── recent " + "─" * max(0, _termwidth() - 10), color)] if heading else []
    out.extend(_dimc("  " + fmt_event(h, snap["now"]), color) for h in hist)
    return out


def render(snap, color=False, mode="classic"):
    views = per_card(snap)
    if mode == "cards":
        return _render_cards(snap, views, color)
    if mode == "bars":
        return _render_bars(snap, views, color)
    if mode == "list":
        return _render_list(snap, views, color)
    out = []
    for view in views:
        out.extend(_render_card(view, color))
    out.extend(_dimc("  " + fmt_event(h, snap["now"]), color) for h in snap.get("history", [])[-5:])
    return "\n".join(out)


def _render_card(view, color):
    """Classic: a header line, the entries, then warnings — no extra separation."""
    return [header(view, view["now"])] + _body_lines(view, color)


def _render_cards(snap, views, color):
    """Each GPU as a colour-ruled card, a blank line between, recent log pulled out below."""
    w = _termwidth()
    out = []
    for idx, view in enumerate(views):
        code = PALETTE[idx % len(PALETTE)]
        title = header(view, view["now"])
        you = " ◀ YOU" if _is_current(view) else ""
        fill = "─" * max(1, w - 3 - len(title) - 1 - len(you))
        out.append(_sgr("╭─ %s %s%s" % (title, fill, you), color, 1, code))
        out.extend("  " + ln for ln in _body_lines(view, color))
        out.append("")
    out.extend(_history_block(snap, color))
    return "\n".join(out).rstrip("\n")


def _render_bars(snap, views, color):
    """Each GPU gets a coloured left gutter bar, a blank line between."""
    out = []
    for idx, view in enumerate(views):
        code = PALETTE[idx % len(PALETTE)]
        bar = _sgr("▌", color, code)
        title = header(view, view["now"])
        you = _sgr(" ◀ YOU", color, 1, 7) if _is_current(view) else ""
        out.append(bar + " " + _sgr(title, color, 1, code) + you)
        out.extend(bar + "   " + ln for ln in _body_lines(view, color))
        out.append("")
    out.extend(_history_block(snap, color))
    return "\n".join(out).rstrip("\n")


def _render_list(snap, views, color):
    """One compact line per GPU: a coloured dot, name, mem, state; your card gets an arrow."""
    out = ["GPUs"]
    for idx, view in enumerate(views):
        code = PALETTE[idx % len(PALETTE)]
        cur = _is_current(view)
        dot = _sgr("●", color, code)
        where, card = view["host"], view.get("card")
        if card:
            gib = ((view.get("cards") or {}).get(card) or {}).get("mib")
            name, mem = "%s %s" % (where, card), ("%dG" % round(gib / 1024) if gib else "")
        else:
            name, mem = where, ""
        q = view["entries"]
        r = running(q)
        state = ("running %s" % r["owner"][:12]) if r else ("%d in line" % len(q) if q else "idle")
        if view.get("paused"):
            state += " · paused"
        extra = (warnings(view) or [""])[0]
        you = _sgr(" ← you", color, 1, 7) if cur else ""
        line = "%s %s %-15s %-4s %-16s %s" % ("▶" if cur else " ", dot, name, mem, state, extra)
        out.append(line.rstrip() + you)
    hist = _history_block(snap, color, heading=False)
    if hist:
        out.append("")
        out.append(_dimc("recent", color))
        out.extend(hist)
    return "\n".join(out)


def warnings(snap):
    out = []
    loaded = snap.get("loaded")
    if loaded is not None:
        text = "GPU in use: %s" % (", ".join(loaded) or "nothing over 1 GB")
        if snap.get("busy") is not None:
            text += " · busy %d%%" % snap["busy"]
        if snap.get("residents"):
            text += " · residents: %s" % ", ".join(snap["residents"])
        out.append(text)
        if loaded and not running(snap["entries"]) and not yielding(snap["entries"]):
            out.append("⚠ unqueued GPU use: something holds the GPU but nothing in the queue is running")
    for m in snap.get("misplaced") or []:
        out.append("⚠ resident on the wrong card: %s" % m)
    for y in yielding(snap["entries"]):
        out.append("⚠ %s was asked to yield %s ago and hasn't stopped; nobody may start until it does"
                   % (y["owner"], fmt_dur(snap["now"] - y["yield_since"])))
    if snap.get("foreign_lock"):
        out.append("⚠ lock file held outside gpuq: %s" % " ".join(snap["foreign_lock"].split()))
    return out


def fmt_event(h, now):
    return "%s %-8s %s %s%s" % (fmt_clock(h["t"], now), h["event"], h["owner"], h["job"][:28],
                                " — " + h["note"] if h.get("note") else "")


def tui_lines(views, sel_host, sel_row, now, width=80):
    """Screen content for the TUI as [(text, style)]. One roomy, colour-coded block per GPU,
    showing its full listing: a coloured header (the focused host's style ends in "s", and your
    own card is tagged ◀ YOU), what's on the card, then every queued job with its models, timing,
    progress, id (when selected) and why — nothing truncated away. A blank line separates GPUs.
    Styles: run / sel / warn / dim / head / "" / "h<idx>" (+ "s" when focused).
    views = [(host, snap or None, error or None)]."""
    L, selected = [], None
    for hi, (host, snap, err) in enumerate(views):
        idx = hi % 6
        focus = hi == sel_host
        mark = "▸ " if focus else "  "       # which host the keys act on
        hstyle = "h%ds" % idx if focus else "h%d" % idx
        if err or snap is None:
            L.append((mark + (err or host + ": no data"), hstyle))
            L.append(("", ""))
            continue
        q = snap["entries"]
        you = "   ◀ YOU" if _is_current(snap) else ""
        base = header(snap, now)
        rule = "─" * max(2, width - len(mark) - len(base) - len(you) - 2)
        L.append((mark + base + " " + rule + you, hstyle))
        for w in warnings(snap):
            L.append(("    " + w, "warn" if w.startswith("⚠") else "dim"))
        if not q:
            L.append(("    queue empty", "dim"))
        for i, (e, timing) in enumerate(entry_timings(q, now)):
            run, yld = e["state"] == "running", e["state"] == "yielding"
            sel = focus and i == sel_row
            if sel:
                selected = (snap, e)
            prim = "%2d %s%s %s" % (i + 1, "▶" if run else "⏏" if yld else " ",
                                    "★" if e.get("important") else " ", e["owner"])
            if e["job"]:
                prim += "  —  " + e["job"]
            L.append((prim, "sel" if sel else "run" if run else "warn" if yld else ""))
            det = "       %s · %s" % (",".join(e["models"]) or "-", timing)
            if e["state"] in ("running", "yielding"):
                det += " · beat %s ago" % fmt_dur(now - (e["heartbeat"] or e["started"]))
            if e.get("progress"):
                det += " · " + e["progress"]
            L.append((det, "dim"))
            if sel:
                bits = ["id %s" % e["id"], "joined %s" % fmt_clock(e["joined"], now)]
                if e.get("max_min"):
                    bits.append("max %s" % fmt_dur(e["max_min"] * 60))
                L.append(("       " + " · ".join(bits), "dim"))
            if e.get("why"):
                L.append(("       why: %s" % e["why"], "dim"))
        L.append(("", ""))
    foc = views[sel_host] if 0 <= sel_host < len(views) else None
    hist = (foc[1] or {}).get("history") if foc and foc[1] else None
    if hist:
        L.append((" recent ", "head"))
        L.extend((fmt_event(h, now), "dim") for h in hist[-4:])
    return L


# ---- I/O: state files, history, compat lock, GPU usage ----

def config():
    try:
        return json.loads(CONFIG.read_text())
    except FileNotFoundError:
        return {}


def remote_cmd():
    """How to invoke gpuq on another host over non-interactive ssh (which often lacks ~/.local/bin
    on PATH). Config `remote_command` overrides."""
    return config().get("remote_command") or "~/.local/bin/gpuq"


def gpu_cards():
    """Every card on this host: NVIDIA (nvidia-smi) first, then AMD (sysfs). See _nvidia_cards,
    amd_cards. GPUQ_CARDS (JSON) overrides all of it, for tests."""
    if os.environ.get("GPUQ_CARDS"):
        return json.loads(os.environ["GPUQ_CARDS"])
    cards = dict(_nvidia_cards())
    for name, c in amd_cards().items():
        while name in cards:
            name += "-%d" % len(cards)
        cards[name] = c
    return cards


def _nvidia_cards():
    """{name: {"uuid", "mib"}} in nvidia-smi order, named by model number ("RTX 3090" -> "3090").
    The last good answer is cached, so a hiccup in nvidia-smi doesn't rename the lines.
    """
    cache = STATE / "cards.json"
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        cards = {}
        for line in r.stdout.splitlines() if r.returncode == 0 else []:
            model, uuid, mib = [x.strip() for x in line.rsplit(",", 2)]
            name = next((t for t in reversed(model.split()) if any(c.isdigit() for c in t)), model)
            if name in cards:
                name += "-%d" % len(cards)
            cards[name] = {"uuid": uuid, "mib": int(float(mib))}
        if cards:
            STATE.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(cards))
            return cards
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    try:
        return json.loads(cache.read_text())
    except (OSError, ValueError):
        return {}


def _size_key(cards):
    """Biggest card, but a discrete one beats an integrated GPU's shared memory."""
    return lambda n: (not cards[n].get("integrated"), cards[n]["mib"])


def default_card(cards, lines=()):
    """The config's default, else the card with the most memory, else an existing line."""
    d = config().get("default")
    if d and (d in cards or d in lines):
        return d
    if cards:
        return max(cards, key=_size_key(cards))
    return next(iter(lines), "gpu")


@contextmanager
def store(write=True):
    """Yields {"lines": {card: [entries]}} — every card's line, the default card first.
    A plain list (one GPU, before 2026-10-28) is read as the default card's line."""
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "queue.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        path = STATE / "queue.json"
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError:
            raw = []
        cards = gpu_cards()
        old = raw if isinstance(raw, list) else None
        found = {} if old is not None else raw.get("lines", {})
        default = default_card(cards, found)
        lines = {default: old if old is not None else found.get(default, [])}
        for name in list(cards) + list(found):
            lines.setdefault(name, found.get(name, []))
        q = {"lines": lines, "default": default, "cards": cards}

        def save():
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"lines": q["lines"]}, indent=1))
            os.replace(tmp, path)
        try:
            yield q
        except QueueError as err:
            if write and err.save:
                save()
            raise
        if write:
            save()


def paused_state():
    try:
        return json.loads((STATE / "paused.json").read_text())
    except FileNotFoundError:
        return None


def set_paused(now, on):
    path = STATE / "paused.json"
    if on:
        path.write_text(json.dumps({"since": now}))
    elif path.exists():
        path.unlink()


def read_evac():
    """The current evac record ({card, to, units, since}) or None."""
    try:
        return json.loads((STATE / "evac.json").read_text())
    except (OSError, ValueError):
        return None


QUEUE = {"id": "-", "owner": "queue", "job": ""}   # history subject for queue-wide events


def log(now, event, e, note=""):
    with open(STATE / "history.jsonl", "a") as f:
        f.write(json.dumps({"t": now, "event": event, "id": e["id"], "owner": e["owner"],
                            "job": e["job"], "note": note}) + "\n")


def history(n=5):
    try:
        lines = (STATE / "history.jsonl").read_text().splitlines()
    except FileNotFoundError:
        return []
    return [json.loads(l) for l in lines[-n:] if l.strip()]


def _compat_path():
    p = COMPAT_LOCK if COMPAT_LOCK is not None else config().get("compat_lock", "")
    return os.path.expanduser(p) if p else ""


def _compat_text():
    if not _compat_path():
        return None
    try:
        return Path(_compat_path()).read_text()
    except FileNotFoundError:
        return None


def foreign_compat():
    """The compat lock's text if someone other than gpuq wrote it, else None."""
    t = _compat_text()
    if t is None or any(l.startswith("gpuq=") for l in t.splitlines()):
        return None
    return t


def write_compat(e):
    if not _compat_path():
        return
    until = (time.strftime("%Y-%m-%d %H:%M", time.localtime(e["started"] + e["max_min"] * 60))
             if e.get("max_min") else "see gpuq")
    Path(_compat_path()).write_text("owner=%s\njob=%s\nuntil=%s\ngpuq=%s\n"
                                 % (e["owner"], e["job"], until, e["id"]))


def clear_compat(eid):
    t = _compat_text()
    if t is not None and "gpuq=%s" % eid in t.splitlines():
        Path(_compat_path()).unlink()


# ---- AMD cards: read straight from sysfs and /proc, no vendor tool needed ----

PCI_IDS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids")


def _read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeDecodeError):
        return ""


def _int(path):
    try:
        return int(_read(path), 0)
    except ValueError:
        return 0


def _pci_name(device_id):
    """Marketing name for an AMD (1002) PCI device id from pci.ids, or ''."""
    for f in PCI_IDS:
        vendor = False
        try:
            lines = Path(f).read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.startswith(("\t", "#")) and line.strip():
                vendor = line.startswith("1002 ")
            elif vendor and line.startswith("\t") and not line.startswith("\t\t"):
                code, _, name = line.strip().partition("  ")
                if code.lower() == device_id.lower().replace("0x", ""):
                    return name
    return ""


def _amd_short_name(full, device_id):
    """"Strix [Radeon 880M / 890M]" -> "880M"; "Navi 31 [Radeon RX 7900 XTX]" -> "7900".
    The first number-bearing word inside the brackets. No name: "amd-<device id>"."""
    inner = full[full.find("[") + 1:full.rfind("]")] if "[" in full else full
    for tok in inner.replace("/", " ").split():
        if any(c.isdigit() for c in tok) and tok.upper() not in ("RX",):
            return tok.upper() if tok.endswith(("m", "M")) else tok
    return "amd-" + device_id.lower().replace("0x", "")


def _kfd_indices():
    """{drm render minor: ROCm device index}: GPU nodes (simd_count > 0) of the kfd topology in
    node order — the order ROCR_VISIBLE_DEVICES counts in. Empty without /sys/class/kfd."""
    out, idx = {}, 0
    nodes = SYSFS / "class" / "kfd" / "kfd" / "topology" / "nodes"
    try:
        names = sorted((n for n in nodes.iterdir() if n.name.isdigit()), key=lambda n: int(n.name))
    except OSError:
        return out
    for n in names:
        props = {}
        for line in _read(n / "properties").splitlines():
            k, _, v = line.partition(" ")
            props[k] = v.strip()
        if int(props.get("simd_count") or 0) > 0:
            out[int(props.get("drm_render_minor") or 0)] = idx
            idx += 1
    return out


def amd_cards():
    """{name: {"vendor": "amd", "uuid", "mib", "integrated", "env_var", "env_value", "pci"}} for
    every amdgpu card in sysfs, named by model number ("Radeon 880M" -> "880M").
    An integrated GPU has a tiny dedicated carve-out and uses system RAM (GTT) as its memory:
    its "mib" is that shared memory. "uuid" is the unique id when the card has one, else its PCI
    address. Pinning is ROCR_VISIBLE_DEVICES (the index among ROCm devices, or GPU-<unique id>)."""
    root = SYSFS / "class" / "drm"
    try:
        cards = sorted((c for c in root.iterdir() if c.name[:4] == "card" and c.name[4:].isdigit()),
                       key=lambda c: int(c.name[4:]))
    except OSError:
        return {}
    kfd, out = _kfd_indices(), {}
    for c in cards:
        d = c / "device"
        if _read(d / "vendor").lower() != "0x1002" or "DRIVER=amdgpu" not in _read(d / "uevent"):
            continue
        dev = _read(d / "device")
        vram, gtt = _int(d / "mem_info_vram_total"), _int(d / "mem_info_gtt_total")
        integrated = vram < (2 << 30) and gtt > vram
        pci = next((l.partition("=")[2] for l in _read(d / "uevent").splitlines()
                    if l.startswith("PCI_SLOT_NAME=")), "")
        uid = _read(d / "unique_id").lower().replace("0x", "")
        minor = next((int(r.name[7:]) for r in sorted((d / "drm").glob("renderD*"))), None) \
            if (d / "drm").is_dir() else None
        if uid and set(uid) != {"0"}:
            env_value = "GPU-" + uid.zfill(16)
        elif minor in kfd:
            env_value = str(kfd[minor])
        else:
            env_value = ""
        name = _amd_short_name(_pci_name(dev), dev)
        while name in out:
            name += "-%d" % len(out)
        out[name] = {"vendor": "amd", "uuid": uid and "GPU-" + uid.zfill(16) or pci or c.name,
                     "mib": (gtt if integrated else vram) // (1 << 20), "integrated": integrated,
                     "env_var": "ROCR_VISIBLE_DEVICES", "env_value": env_value, "pci": pci}
    return out


def amd_usage(cards):
    """{card: {"jobs": [...], "busy": percent or None}} for AMD cards. Per-process memory comes from
    the kernel's DRM fdinfo (/proc/PID/fdinfo/*: drm-memory-vram / drm-memory-gtt), which needs no
    amd-smi; you only see processes you may read. Processes under GPU_USER_MIN_MIB are not jobs."""
    amd = {n: c for n, c in cards.items() if c.get("vendor") == "amd"}
    if not amd:
        return {}
    by_pci = {c["pci"]: n for n, c in amd.items() if c.get("pci")}
    seen, per = set(), {n: {} for n in amd}
    try:
        pids = [p for p in PROC.iterdir() if p.name.isdigit()]
    except OSError:
        pids = []
    for p in pids:
        try:
            fds = list((p / "fdinfo").iterdir())
        except OSError:
            continue
        for f in fds:
            try:
                info = f.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            if "drm-driver:\tamdgpu" not in info and "drm-driver: amdgpu" not in info:
                continue
            kv = {}
            for line in info.splitlines():
                k, _, v = line.partition(":")
                kv[k.strip()] = v.strip()
            card, cid = by_pci.get(kv.get("drm-pdev", "")), (p.name, kv.get("drm-client-id"))
            if card is None or cid in seen:
                continue
            seen.add(cid)
            kib = 0
            for key in ("drm-memory-vram", "drm-memory-gtt"):
                try:
                    kib += int(kv.get(key, "0").split()[0])
                except (ValueError, IndexError):
                    pass
            if amd[card]["integrated"] or kib:
                name = _gpu_proc_name(_exe(p) or _read(p / "comm") or p.name)
                per[card][name] = per[card].get(name, 0) + kib
    use = {}
    for n, c in amd.items():
        jobs = ["%s %.1fG" % (proc, kib / (1 << 20)) for proc, kib in sorted(per[n].items(), key=lambda x: -x[1])
                if kib >= GPU_USER_MIN_MIB * 1024]
        busy = _read(SYSFS / "class" / "drm" / _card_dir(c) / "device" / "gpu_busy_percent")
        use[n] = {"jobs": jobs, "busy": int(busy) if busy.isdigit() else None}
    return use


def _card_dir(c):
    """The drm/cardN directory name for an AMD card, found again by its PCI address."""
    root = SYSFS / "class" / "drm"
    for d in root.glob("card[0-9]*"):
        if f"PCI_SLOT_NAME={c.get('pci')}" in _read(d / "device" / "uevent"):
            return d.name
    return "card?"


def _exe(p):
    try:
        return os.readlink(p / "exe")
    except OSError:
        return ""


GPU_USER_MIN_MIB = 1024  # below this it's desktop apps, whisper, a slicer: not a job


def _gpu_proc_name(path):
    parts = path.split("/")
    for marker in ("venv", ".venv"):
        if marker in parts and parts.index(marker) > 0:
            return parts[parts.index(marker) - 1]  # /…/project/venv/bin/python -> project
    return parts[-1].split()[0]


def gpu_loaded():
    """Big GPU users per nvidia-smi, e.g. ["ollama 7.6G"]. None = unknown, and the line is not shown.
    Not `ollama ps`: that only sees one server and says "nothing" while other jobs hold the card.
    Better to not know what is loaded than to be told nothing is when something is."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    out = []
    for line in r.stdout.splitlines():
        name, _, mib = line.rpartition(",")
        try:
            mib = int(mib.strip())
        except ValueError:
            return None
        if mib >= GPU_USER_MIN_MIB:
            out.append("%s %.1fG" % (_gpu_proc_name(name.strip()), mib / 1024))
    return out


def card_usage(cards, residents):
    """{card: {"jobs": ["ollama 7.6G"], "residents": [...], "misplaced": [...]}} per nvidia-smi.
    residents = {card: [process names]} from the config. None = unknown."""
    if os.environ.get("GPUQ_CARDS"):
        return None
    amd = amd_usage(cards)
    nv_cards = {n: c for n, c in cards.items() if c.get("vendor") != "amd"}
    if amd and not nv_cards:
        use = {n: {"jobs": u["jobs"], "residents": [], "misplaced": [], "busy": u["busy"]}
               for n, u in amd.items()}
        _place_residents(use, residents)
        return use
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    by_uuid = {c["uuid"]: name for name, c in nv_cards.items()}
    home = {p: card for card, procs in residents.items() for p in procs}
    use = {name: {"jobs": [], "residents": [], "misplaced": []} for name in nv_cards}
    for line in r.stdout.splitlines():
        uuid, _, rest = line.partition(",")
        path, _, mib = rest.rpartition(",")
        try:
            mib = int(mib.strip())
        except ValueError:
            return None
        card, proc = by_uuid.get(uuid.strip()), _gpu_proc_name(path.strip())
        if card is None:
            continue
        label = "%s %.1fG" % (proc, mib / 1024)
        if proc in home:
            use[card]["residents" if home[proc] == card else "misplaced"].append(
                label if home[proc] == card else "%s is on the %s, meant for the %s" % (proc, card, home[proc]))
        elif mib >= GPU_USER_MIN_MIB:
            use[card]["jobs"].append(label)
    for n, u in amd.items():
        use[n] = {"jobs": u["jobs"], "residents": [], "misplaced": [], "busy": u["busy"]}
    if amd:
        _place_residents({n: use[n] for n in amd}, residents)
    return use


def _place_residents(use, residents):
    """AMD cards: a job whose process name is a configured resident of some card is a resident
    (on the right card) or misplaced (on another), exactly as for NVIDIA cards."""
    home = {p: card for card, procs in residents.items() for p in procs}
    for card, u in use.items():
        keep = []
        for label in u["jobs"]:
            proc = label.rpartition(" ")[0]
            if proc in home:
                (u["residents"] if home[proc] == card else u["misplaced"]).append(
                    label if home[proc] == card else
                    "%s is on the %s, meant for the %s" % (proc, card, home[proc]))
            else:
                keep.append(label)
        u["jobs"] = keep


def local_snapshot(now):
    with store(write=False) as Q:
        lines, cards = Q["lines"], Q["cards"]
        use = card_usage(cards, config().get("residents", {}))
        has_amd = any(c.get("vendor") == "amd" for c in cards.values())
        if use is None or (len(cards) < 2 and not has_amd):
            loaded = gpu_loaded()
        else:
            loaded = [j for u in use.values() for j in u["jobs"]]
        return snapshot(lines[Q["default"]], now, history(), loaded, paused=paused_state(),
                        lines=lines, cards=cards, card_use=use, default=Q["default"], evac=read_evac())


def remote_snapshot(host):
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                            remote_cmd() + " show --json"],
                           capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None, "%s: no answer within 30 s" % host
    if r.returncode != 0:
        return None, "%s: unreachable (%s)" % (host, (r.stderr.strip().splitlines() or ["?"])[-1])
    return json.loads(r.stdout), None


# ---- TUI ----

def _self(*args):
    """Command line that re-runs this gpuq (for the local half of `act` and `gpuq run`)."""
    return [sys.executable, "-m", "gpuq", *args]


def _self_env():
    root = str(Path(__file__).resolve().parent.parent)
    return dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])))


def act(host, *args):
    """Run a gpuq command against HOST's queue (here, or over ssh): (exit code, last line)."""
    if host == HOST:
        cmd = _self(*args)
    else:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
               remote_cmd() + " " + shlex.join(args)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                           env=_self_env() if host == HOST else None)
    except subprocess.TimeoutExpired:
        return 1, "%s: no answer within 30 s" % host
    lines = (r.stderr or r.stdout).strip().splitlines()
    return r.returncode, lines[-1] if lines else ""


def fetch_views(hosts):
    """[(host, snapshot of one card's line or None, error or None)] — a view per card per host."""
    views = []
    for h in hosts:
        if h == HOST:
            snap, err = local_snapshot(time.time()), None
        else:
            snap, err = remote_snapshot(h)
        views.extend((h, v, None) for v in per_card(snap)) if snap else views.append((h, None, err))
    return views


TUI_KEYS = " ↑↓ select · k/Shift-↑ move up (sooner) · j/Shift-↓ move down (later) · d remove · p pause/resume · Tab/←→ host · r refresh · q quit "


# Shift-↑/↓ as raw escape sequences, for terminals whose terminfo doesn't decode them
SHIFT_UP, SHIFT_DOWN = -2, -3
SEQ_KEYS = {"[1;2A": SHIFT_UP, "[1;2B": SHIFT_DOWN, "[a": SHIFT_UP, "[b": SHIFT_DOWN}


def _tui(args):
    import curses
    import locale
    locale.setlocale(locale.LC_ALL, "")
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]

    def loop(scr):
        curses.curs_set(0)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        style = {"head": curses.A_REVERSE | curses.A_BOLD, "run": curses.color_pair(1) | curses.A_BOLD,
                 "sel": curses.A_REVERSE, "warn": curses.color_pair(2) | curses.A_BOLD,
                 "dim": curses.A_DIM, "dimhead": curses.A_REVERSE | curses.A_DIM,
                 "": curses.A_NORMAL}
        pal = [curses.COLOR_CYAN, curses.COLOR_MAGENTA, curses.COLOR_YELLOW,
               curses.COLOR_GREEN, curses.COLOR_BLUE, curses.COLOR_RED]
        for i, c in enumerate(pal):
            curses.init_pair(11 + i, c, -1)

        def resolve(st):
            """A fixed style name, or a GPU-coloured header key 'h<idx>' (+ 's' when focused)."""
            if len(st) >= 2 and st[0] == "h" and st[1].isdigit():
                a = curses.color_pair(11 + int(st[1])) | curses.A_BOLD
                return a | curses.A_REVERSE if st.endswith("s") else a
            return style.get(st, curses.A_NORMAL)
        scr.timeout(250)
        views, fetched, sh, sr, msg = fetch_views(hosts), time.time(), 0, 0, ""

        def shift_dir(k):
            """-1 / +1 for a move-up / move-down key (k/K/Shift-↑ toward #1, j/J/Shift-↓ away), else 0."""
            name = curses.keyname(k) if k >= 0 else b""  # kUP2/kDN2: xterm-style terminfo
            if k in (ord("K"), ord("k"), curses.KEY_SR, SHIFT_UP) or name == b"kUP2":
                return -1
            if k in (ord("J"), ord("j"), curses.KEY_SF, SHIFT_DOWN) or name == b"kDN2":
                return 1
            return 0

        def entries():
            snap = views[sh][1]
            return snap["entries"] if snap else []

        def on_card():
            """--gpu for a position-based command in the selected card's line (older hosts: none)."""
            card = (views[sh][1] or {}).get("card")
            return ["--gpu", card] if card else []

        def put(y, text, attr):
            try:
                scr.addnstr(y, 0, text.ljust(w), w - 1, attr)
            except curses.error:
                pass

        while True:
            if time.time() - fetched >= args.every:
                views, fetched = fetch_views(hosts), time.time()
            sr = max(0, min(sr, len(entries()) - 1))
            h, w = scr.getmaxyx()
            scr.erase()
            for y, (text, st) in enumerate(tui_lines(views, sh, sr, time.time(), w)[:h - 1]):
                put(y, text, resolve(st))
            put(h - 1, msg or TUI_KEYS, curses.A_REVERSE)
            scr.refresh()
            k = scr.getch()
            if k == -1:
                continue
            if k == 27:     # a lone Esc quits; an undecoded escape sequence (Shift-↑ etc.) doesn't
                scr.timeout(30)
                seq = ""
                while (c := scr.getch()) != -1:
                    seq += chr(c)
                scr.timeout(250)
                k = SEQ_KEYS.get(seq, -1 if seq else 27)
                if k == -1:
                    continue
            msg = ""
            q, host = entries(), views[sh][0]
            if k in (ord("q"), 27):
                return
            elif k == curses.KEY_UP:
                sr -= 1
            elif k == curses.KEY_DOWN:
                sr += 1
            elif k in (9, curses.KEY_RIGHT, curses.KEY_BTAB, curses.KEY_LEFT):
                step = -1 if k in (curses.KEY_BTAB, curses.KEY_LEFT) else 1
                sh, sr = (sh + step) % len(views), 0
            elif k == ord("r"):
                fetched = 0
            elif k == ord("p") and views[sh][1]:
                verb = "resume" if views[sh][1].get("paused") else "pause"
                code, out = act(host, verb)
                msg = " " + (out or "%s: queue %sd" % (host, verb)) + " "
                fetched = 0
            elif shift_dir(k) and q:
                pos = sr + 1 + shift_dir(k)
                if not 1 <= pos <= len(q):
                    msg = " #%d is already at the %s " % (sr + 1, "top" if pos < 1 else "bottom")
                else:
                    code, out = act(host, "mv", q[sr]["id"], str(pos), *on_card())
                    msg = " " + (out or "moved %s to #%d" % (q[sr]["owner"], pos)) + " "
                    if code == 0:
                        sr = pos - 1
                    fetched = 0
            elif k == ord("d") and q:
                e = q[sr]
                put(h - 1, " remove #%d %s (%s)? This frees its place, it does not stop the job. y/n "
                    % (sr + 1, e["owner"], e["job"]), curses.A_REVERSE | curses.A_BOLD)
                scr.refresh()
                scr.timeout(-1)
                yes = scr.getch() in (ord("y"), ord("Y"))
                scr.timeout(250)
                if yes:
                    code, out = act(host, "rm", e["id"])
                    msg = " " + (out or "removed %s" % e["owner"]) + " "
                    fetched = 0

    curses.wrapper(loop)
    return 0


# ---- CLI ----

def _forward(argv, host):
    """Re-run this command on HOST, minus the --host option."""
    rest, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == "--host":
            skip = True
        elif not a.startswith("--host="):
            rest.append(a)
    cmd = remote_cmd() + (" " + shlex.join(rest) if rest else "")
    return subprocess.call(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, cmd])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="gpuq", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", help="run the command against HOST's queue over ssh")
    sub = ap.add_subparsers(dest="cmd")
    card = argparse.ArgumentParser(add_help=False)
    card.add_argument("--gpu", help="which card's line, by model number (e.g. 3060); see `gpuq show`")
    view = argparse.ArgumentParser(add_help=False)   # layout flag, shared by show/watch/tui
    view.add_argument("--mode", choices=["classic", "cards", "bars", "list"], default="classic",
                      help="layout: classic (default), cards, bars, or list")
    hosts_default = config().get("hosts") or HOST   # config "hosts": comma list; default = this host
    p = sub.add_parser("show", help="print the queue once", parents=[view])
    p.add_argument("--json", action="store_true")
    p.add_argument("--hosts", default=hosts_default,
                   help="comma-separated hosts (default: %s); --json is always local only"
                        % hosts_default)
    p = sub.add_parser("watch", help="live view, refreshed", parents=[view])
    p.add_argument("--hosts", default=hosts_default, help="comma-separated (default: %s)" % hosts_default)
    p.add_argument("--every", type=float, default=5, help="seconds between refreshes")
    p = sub.add_parser("tui", help="interactive view: select, reorder, remove")
    p.add_argument("--hosts", default=hosts_default, help="comma-separated (default: %s)" % hosts_default)
    p.add_argument("--every", type=float, default=5, help="seconds between refreshes")
    p = sub.add_parser("join", help="get in line; prints your id", parents=[card])
    p.add_argument("--owner", required=True)
    p.add_argument("--job", required=True,
                   help="what the job is — be descriptive; the tui shows it in full, short views trim")
    p.add_argument("--why", default="",
                   help="why it's running / who it's for — a full phrase is fine; shown in full in the tui")
    p.add_argument("--models", default="", help="comma-separated")
    p.add_argument("--est", type=int, help="expected minutes")
    p.add_argument("--max", type=int, help="minutes after which the sweep purges you")
    p.add_argument("--important", action="store_true",
                   help="never jumped by short jobs (default: a job of hours goes ahead of one of days)")
    p = sub.add_parser("edit", help="change an entry's details; id and place are kept"
                       " (--gpu moves a waiting entry to the end of that card's line)", parents=[card])
    p.add_argument("id")
    for f in ("--job", "--why", "--models"):
        p.add_argument(f)
    p.add_argument("--est", type=int)
    p.add_argument("--max", type=int)
    p.add_argument("--important", action=argparse.BooleanOptionalAction, default=None)
    p = sub.add_parser("start", help="take the GPU (only as #1)")
    p.add_argument("id")
    p.add_argument("--adopt", action="store_true",
                   help="take over a lock file written outside gpuq (backfilling a live job)")
    p = sub.add_parser("beat", help="heartbeat; at least every 15 min (exit 5 = yield: stop, poll start)")
    p.add_argument("id")
    p.add_argument("--progress")
    p = sub.add_parser("done", help="release the GPU and leave the line")
    p.add_argument("id")
    p = sub.add_parser("env", help="print the job's card as CUDA_VISIBLE_DEVICES=GPU-<uuid>")
    p.add_argument("id")
    p = sub.add_parser("mv", help="move an entry (position or id) to a position", parents=[card])
    p.add_argument("src")
    p.add_argument("dst")
    p = sub.add_parser("run", help="hold the GPU for CMD: start, beat, stop on yield, rerun at #1, done",
                       usage="gpuq run ID [--every S] [--grace S] [--on-stop SHELL] [--progress-file F] -- CMD ...",
                       description="Runs CMD in its own process group once ID may start, beats every"
                       " --every seconds, stops it on a yield (beat exit 5) and reruns it from the"
                       " start when ID is #1 again — so CMD must be safe to rerun. Calls done when"
                       " CMD ends, or when this wrapper is interrupted. Exits with CMD's code.")
    p.add_argument("id")
    p.add_argument("--every", type=float, default=60, help="seconds between beats / start polls")
    p.add_argument("--grace", type=float, default=30, help="seconds after SIGTERM before SIGKILL")
    p.add_argument("--on-stop", help="shell command run after CMD is stopped (e.g. unload a model)")
    p.add_argument("--progress-file", help="its last line is sent as --progress with each beat")
    p = sub.add_parser("yield", help="ask the running job to step back (default to #2)")
    p.add_argument("id")
    p.add_argument("to", nargs="?")
    p = sub.add_parser("rm", help="remove an entry (position or id)", parents=[card])
    p.add_argument("ref")
    sub.add_parser("pause", help="hold the line: nobody may start until resume")
    sub.add_parser("resume", help="release the line; #1 gets a fresh 15 minutes")
    sub.add_parser("sweep", help="garbage collection (the timer runs this)")
    p = sub.add_parser("evac", help="move a card's resident services off it, to free the whole card")
    p.add_argument("card", nargs="?", help="card to empty (default: the config default card)")
    p.add_argument("--to", help="card to move residents to (default: the other card)")
    p.add_argument("--dry-run", action="store_true", help="show what would move; change nothing")
    p = sub.add_parser("restore", help="undo an evac: put resident services back on their home card")
    p.add_argument("card", nargs="?")
    p.add_argument("--dry-run", action="store_true", help="show what would change; change nothing")
    # everything after the first "--" is gpuq run's command, never gpuq's own options
    head, run_cmd = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    args = ap.parse_args(head)
    args.run_cmd = run_cmd

    if args.host and args.host != HOST:
        return _forward(argv, args.host)

    now = float(os.environ.get("GPUQ_NOW") or time.time())
    try:
        return _run(args, now)
    except QueueError as err:
        print("gpuq: %s" % err, file=sys.stderr)
        return err.code


def _run(args, now):
    cmd = args.cmd or "show"
    if cmd == "show":
        if getattr(args, "json", False):
            print(json.dumps(local_snapshot(now)))    # machine-readable: always local, never recurses
            return 0
        color, mode = sys.stdout.isatty(), getattr(args, "mode", None) or "classic"
        # bare `gpuq` has no show-subparser args, so fall back to the same default
        hosts_str = getattr(args, "hosts", None) or config().get("hosts") or HOST
        hosts = [h.strip() for h in hosts_str.split(",") if h.strip()] or [HOST]
        if hosts == [HOST]:
            print(render(local_snapshot(now), color, mode))
            return 0
        blocks = []
        for h in hosts:
            if h == HOST:
                blocks.append(render(local_snapshot(now), color, mode))
            else:
                snap, err = remote_snapshot(h)
                if snap:
                    snap["now"] = now   # age the remote queue against our clock
                blocks.append(err or render(snap, color, mode))
        print("\n\n".join(blocks))
        return 0
    if cmd == "watch":
        return _watch(args)
    if cmd == "tui":
        return _tui(args)
    if cmd == "run":
        return _hold(args)
    if cmd == "evac":
        return _evac(args, now)
    if cmd == "restore":
        return _restore(args, now)
    with store() as Q:
        lines, cards = Q["lines"], Q["cards"]
        multi = len(lines) > 1

        def line_named(name):
            if name not in lines:
                raise QueueError(2, "no card %r here — cards: %s" % (name, ", ".join(lines)))
            return lines[name]

        def line_of(ref, by_id=False):
            """The line holding id REF; a position (unless by_id) counts in --gpu's line, or the default's."""
            for q in lines.values():
                if any(e["id"] == ref for e in q):
                    return q
            if by_id:
                raise QueueError(4, "no entry %r in the queue" % ref)
            return line_named(getattr(args, "gpu", None) or Q["default"])

        def where(q):
            return " on the %s" % next(n for n, x in lines.items() if x is q) if multi else ""

        if cmd == "join":
            models = [m.strip() for m in args.models.split(",") if m.strip()]
            name = args.gpu or config().get("owners", {}).get(args.owner) or Q["default"]
            q = line_named(name)
            taken = {x["id"] for other in lines.values() for x in other}
            e = join(q, now, args.owner, args.job, args.why, models, args.est, args.max,
                     bool(args.important), taken=taken)
            e["gpu"] = name
            log(now, "joined", e, "#%d%s" % (len(q), where(q)))
            _jump(q, now, e)
            print(e["id"])
        elif cmd == "start":
            q = line_of(args.id, by_id=True)
            if paused_state() and q[find(q, args.id, by_id=True)]["state"] == "waiting":
                raise QueueError(3, "the queue is paused — retry until it resumes")
            foreign = foreign_compat()
            if foreign and not args.adopt:
                raise QueueError(3, "lock file held outside gpuq: %s" % " ".join(foreign.split()))
            was = q[find(q, args.id, by_id=True)]["state"]
            try:
                e = start(q, now, args.id)
            except QueueError as err:
                if was == "yielding" and err.save:
                    e = q[find(q, args.id, by_id=True)]
                    clear_compat(e["id"])
                    log(now, "yielded", e, "now #%d" % (q.index(e) + 1))
                raise
            write_compat(e)
            if was != "running":
                log(now, "started", e, "adopted a live job" if args.adopt else where(q).strip())
            if multi and card_env(Q, args.id):
                print(card_env(Q, args.id))
        elif cmd == "env":
            line_of(args.id, by_id=True)
            print(card_env(Q, args.id) or "")
        elif cmd == "edit":
            q = line_of(args.id)
            e = q[find(q, args.id)]
            for key, val in (("job", args.job), ("why", args.why), ("est_min", args.est),
                             ("max_min", args.max)):
                if val is not None:
                    e[key] = val
            if args.models is not None:
                e["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
            if args.important is not None:
                e["important"] = args.important
            if args.gpu and lines.get(args.gpu) is not q:
                if e["state"] != "waiting":
                    raise QueueError(2, "%s is %s — it can change cards only while waiting"
                                     % (e["id"], e["state"]))
                dest = line_named(args.gpu)
                remove(q, now, e["id"])
                dest.append(e)
                e["gpu"], e["skips"] = args.gpu, 0
                normalize(dest, now)
                q = dest
            log(now, "edited", e, where(q).strip())
            if args.est is not None or args.important is not None or args.gpu:
                _jump(q, now, e)
        elif cmd == "beat":
            beat(line_of(args.id, by_id=True), now, args.id, args.progress)
        elif cmd == "done":
            e = done(line_of(args.id, by_id=True), now, args.id)
            clear_compat(e["id"])
            log(now, "done", e)
        elif cmd in ("mv", "yield"):
            if cmd == "yield":
                q = line_of(args.id, by_id=True)
                if len(q) < 2:
                    raise QueueError(2, "nobody to yield to — the borrower must join first")
                src, dst = args.id, args.to or "2"
            else:
                src, dst = args.src, args.dst
                q = line_of(src)
            before = {x["id"]: x["state"] for x in q}
            e = move(q, now, src, dst)
            log(now, "moved", e, "to #%s%s" % (dst, where(q)))
            _note_yields(q, now, before)
        elif cmd == "rm":
            q = line_of(args.ref)
            e = remove(q, now, args.ref)
            clear_compat(e["id"])
            log(now, "removed", e, "by hand")
        elif cmd in ("pause", "resume"):
            set_paused(now, cmd == "pause")
            log(now, cmd + "d", QUEUE)
            if cmd == "resume":
                for q in lines.values():
                    normalize(q, now)
                    if q and q[0]["state"] == "waiting":
                        q[0]["head_since"] = now     # a fresh 15 minutes for #1
        elif cmd == "sweep":
            for q in lines.values():
                for event, e, note in sweep(q, now, paused=bool(paused_state())):
                    clear_compat(e["id"])
                    log(now, event, e, note)
                    print("%s %s (%s): %s" % (event, e["owner"], e["id"], note))
    return 0


def card_env(Q, eid):
    """"CUDA_VISIBLE_DEVICES=GPU-<uuid>" for the card whose line holds EID, or None if unknown."""
    for name, q in Q["lines"].items():
        if any(e["id"] == eid for e in q):
            c = Q["cards"].get(name) or {}
            if c.get("env_var"):        # AMD: ROCR_VISIBLE_DEVICES=<index or GPU-<unique id>>
                return "%s=%s" % (c["env_var"], c["env_value"]) if c.get("env_value") else None
            uuid = c.get("uuid")
            return "CUDA_VISIBLE_DEVICES=%s" % uuid if uuid else None
    return None


def _hold(args):
    """gpuq run: hold the GPU for a command — start, beat, yield, resume, done."""
    cmd = args.run_cmd
    if not cmd:
        raise QueueError(2, "nothing to run: gpuq run ID [options] -- CMD ...")

    def say(text):
        print("gpuq run %s: %s" % (args.id, text), file=sys.stderr, flush=True)

    def gq(*a):
        return subprocess.run(_self(*a), capture_output=True, text=True, env=_self_env()).returncode

    def stop(proc):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=args.grace)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            proc.wait()
        if args.on_stop:
            subprocess.run(args.on_stop, shell=True)

    def progress():
        try:
            lines = Path(args.progress_file).read_text().strip().splitlines()
            return lines[-1][:120] if lines else None
        except (OSError, TypeError):
            return None

    proc = None

    def on_signal(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_signal)
    try:
        while True:
            while (rc := gq("start", args.id)) == 3:     # 3 also acknowledges a yield
                time.sleep(args.every)
            if rc != 0:
                say("start exited %d — not running" % rc)
                return rc
            env = dict(os.environ)
            with store(write=False) as Q:
                pin = card_env(Q, args.id) if len(Q["lines"]) > 1 else None
            if pin:
                env[pin.partition("=")[0]] = pin.partition("=")[2]
            say("granted%s, running: %s" % (" (%s)" % pin if pin else "", shlex.join(cmd)))
            proc = subprocess.Popen(cmd, start_new_session=True, env=env)
            while True:
                try:
                    proc.wait(timeout=args.every)
                    break
                except subprocess.TimeoutExpired:
                    pass
                p = progress()
                rc = gq("beat", args.id, *(["--progress", p] if p else []))
                if rc in (4, 5):
                    break
            if proc.returncode is not None:
                code, proc = proc.returncode, None
                gq("done", args.id)
                say("finished (exit %d), released the GPU" % code)
                return code
            stop(proc)
            proc = None
            if rc == 4:
                say("purged from the queue — stopped")
                return 4
            say("yielded — stopped, waiting for the GPU again")
    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            stop(proc)
        gq("done", args.id)
        say("interrupted — stopped and left the line")
        return 130


def _note_yields(q, now, before):
    """Log (and tell stderr about) any runner that a reorder just asked to yield."""
    for x in q:
        if x["state"] == "yielding" and before.get(x["id"]) == "running":
            log(now, "yield?", x, "asked to step back to #%d" % (q.index(x) + 1))
            print("asked %s to yield: it stops at its next heartbeat, then waits at #%d"
                  % (x["owner"], q.index(x) + 1), file=sys.stderr)


def _jump(q, now, e):
    before = {x["id"]: x["state"] for x in q}
    passed = jump(q, now, e["id"])
    if passed:
        note = "short job: %s ahead of %s" % (fmt_dur(remaining(e, now)), ", ".join(
            "%s (~%s left)" % (x["owner"], fmt_dur(remaining(x, now))) for x in passed))
        log(now, "jumped", e, note)
        print("jumped to #%d — %s" % (q.index(e) + 1, note), file=sys.stderr)
        _note_yields(q, now, before)


# ---- evacuate / restore: move a card's resident services off it, to free the whole card ----

def _unit_for(proc, cfg):
    """The systemd user unit for a resident process name (config `units`, else <proc>.service)."""
    return (cfg.get("units") or {}).get(proc) or (proc + ".service")


def _evac_dropin(unit):
    return Path.home() / ".config" / "systemd" / "user" / (unit + ".d") / "gpuq-evac.conf"


def evac_targets(cards, cfg, card=None, to=None):
    """(card, to, uuid, units) for emptying CARD onto TO. Defaults: CARD = config default or the
    biggest card; TO = the biggest of the rest. Raises QueueError on a missing/unknown card."""
    card = card or cfg.get("default") or (max(cards, key=_size_key(cards)) if cards else None)
    if card not in cards:
        raise QueueError(2, "unknown card %r — cards: %s" % (card, ", ".join(cards) or "none"))
    others = [n for n in cards if n != card]
    to = to or (max(others, key=_size_key(cards)) if others else None)
    if to not in cards:
        raise QueueError(2, "no second card to move residents to")
    units = [_unit_for(p, cfg) for p in (cfg.get("residents") or {}).get(card, [])]
    if not units:
        raise QueueError(2, "no resident services configured on the %s (see the config file, 'residents')"
                         % card)
    return card, to, cards[to].get("env_value") or cards[to]["uuid"], units


def _systemctl(*a):
    return subprocess.run(["systemctl", "--user", *a], capture_output=True, text=True)


def _restart_all(units):
    """daemon-reload, then restart each unit; returns 0, or 1 if any restart failed (printed)."""
    _systemctl("daemon-reload")
    rc = 0
    for u in units:
        r = _systemctl("restart", u)
        if r.returncode != 0:
            rc = 1
            print("  ! restart %s failed: %s" % (u, (r.stderr or "").strip()), file=sys.stderr)
    return rc


def _evac(args, now):
    cards = gpu_cards()
    card, to, uuid, units = evac_targets(cards, config(), args.card, args.to)
    var = cards[to].get("env_var", "CUDA_VISIBLE_DEVICES")
    print("evac: empty the %s — move residents to the %s (%s)" % (card, to, uuid))
    for u in units:
        print("  %s → %s=%s, then restart" % (u, var, uuid))
    if args.dry_run:
        print("(dry run — nothing changed)")
        return 0
    body = "[Service]\nEnvironment=%s=%s\n" % (var, uuid)
    for u in units:
        d = _evac_dropin(u)
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(body)
    rc = _restart_all(units)
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "evac.json").write_text(json.dumps({"card": card, "to": to, "units": units, "since": now}))
    print("the %s is clear — `gpuq restore` to put them back" % card)
    return rc


def _restore(args, now):
    st = read_evac()
    cfg = config()
    card = args.card or (st or {}).get("card") or cfg.get("default")
    units = (st or {}).get("units") or [_unit_for(p, cfg)
                                        for p in (cfg.get("residents") or {}).get(card, [])]
    if not units:
        raise QueueError(2, "nothing to restore")
    print("restore: put residents back on the %s" % card)
    for u in units:
        print("  %s → drop the override, then restart" % u)
    if args.dry_run:
        print("(dry run — nothing changed)")
        return 0
    for u in units:
        d = _evac_dropin(u)
        try:
            d.unlink()
            if not any(d.parent.iterdir()):
                d.parent.rmdir()
        except OSError:
            pass
    rc = _restart_all(units)
    try:
        (STATE / "evac.json").unlink()
    except OSError:
        pass
    print("residents back on the %s" % card)
    return rc


def _watch(args):
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    color = sys.stdout.isatty()
    mode = getattr(args, "mode", None) or "classic"
    try:
        while True:
            now = time.time()
            blocks = []
            for h in hosts:
                if h == HOST:
                    blocks.append(render(local_snapshot(now), color, mode))
                else:
                    snap, err = remote_snapshot(h)
                    if snap:
                        snap["now"] = now   # render remote ages against our clock
                    blocks.append(err or render(snap, color, mode))
            sys.stdout.write("\033[H\033[2J" + "\n\n".join(blocks) + "\n\n(ctrl-c to quit)\n")
            sys.stdout.flush()
            time.sleep(args.every)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())