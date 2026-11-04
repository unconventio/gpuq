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
Humans:     gpuq                          the live view (TUI) at a terminal; piped, prints the line once
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

Fleet:      gpuq inventory [--hosts a,b]    what each card offers: usable memory, tags, residents, wait
            gpuq where --vram 16 [--tag rocm] [--est 30]   which card on which host fits, soonest first
            gpuq join ... --vram 16 [--tag T]              picks the best local card that fits
            gpuq free [CARD]  ·  gpuq restore             unload/stop a card's residents; bring them back
            gpuq init                                      write a starter config from what is detected
            gpuq mcp                                       the same, as an MCP server for agents (stdio)
JSON:       --json on join/start/beat/done/... prints one object (state go/wait/gone, ok/yield/purged);
            `gpuq join --spec -` reads the job as a JSON object.

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
from .core import QueueError, fmt_dur, yielding, running, normalize, find, remove, prio, prio_mark, place, PRIORITY
from . import fleet

HOST = socket.gethostname()
STATE = Path(os.environ.get("GPUQ_STATE") or Path.home() / ".local" / "state" / "gpuq")
SYSFS = Path(os.environ.get("GPUQ_SYSFS") or "/sys")     # roots are overridable so tests can
PROC = Path(os.environ.get("GPUQ_PROC") or "/proc")      # use fixture trees
CONFIG = Path(os.environ.get("GPUQ_CONFIG") or Path.home() / ".config" / "gpuq" / "config.json")
# Optional lock file for tools that predate gpuq: written while a job runs, removed when it ends
# (only if gpuq wrote it). Set `compat_lock` in the config, or $GPUQ_COMPAT_LOCK. "" disables.
COMPAT_LOCK = os.environ.get("GPUQ_COMPAT_LOCK")     # None = ask the config
# Per-host settings (all optional), see examples/config.json:
#   default, owners, cards, residents, units, hosts, compat_lock, remote_command
# Residents are always-on services that live on a card outside the queue: never "unqueued use",
# flagged if on the wrong card. `cards` caps, reserves and tags a card's memory (see fleet.py).


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
    cards = snap.get("cards") or {}
    card = snap.get("card") or (next(iter(cards)) if len(cards) == 1 else None)
    if card:
        c = cards.get(card) or {}
        gib = c.get("mib")
        where = "%s%s · %s" % (c.get("model") or card,
                               " (%d GB%s)" % (round(gib / 1024), " shared" if c.get("integrated") else "")
                               if gib else "", snap["host"])
    # card and machine, then the running job in brief: who/what, model, ETA; its id goes last
    r = q[0] if q and q[0]["state"] in ("running", "yielding") else None
    if r:
        text = "%s · %s %s: %s" % (where, "▶" if r["state"] == "running" else "⏏",
                                   r["owner"][:12], _trim(r["job"], 28))
        if r["models"]:
            text += " · " + ",".join(r["models"])
        if r.get("est_min") and r.get("started"):
            end = r["started"] + r["est_min"] * 60
            text += (" · ends ~%s" % fmt_clock(end, now) if end > now
                     else " · %s over its estimate" % fmt_dur(now - end))
        if len(q) > 1:
            text += " · +%d waiting" % (len(q) - 1)
    else:
        text = "%s · %s" % (where, "%d in line" % len(q) if q else "queue empty")
    if snap.get("paused"):
        text += " · ⏸ PAUSED since %s — nobody may start" % fmt_clock(snap["paused"]["since"], now)
    ev = snap.get("evac")
    if ev and ev.get("card") == snap.get("card"):
        text += " · ⏏ EVACUATED → %s" % ev.get("to")
    if r:
        text += " · %s" % r["id"]
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
                over = elapsed - e["est_min"] * 60
                t = "%s of ~%s, %s" % (fmt_dur(elapsed), fmt_dur(e["est_min"] * 60),
                                       "%s over, end unknown" % fmt_dur(over) if over > 0
                                       else "ends ~%s" % fmt_clock(cursor, now))
            else:
                cursor, t = None, "%s so far, no estimate" % fmt_dur(elapsed)
        elif e["state"] == "yielding":
            t = "yielding, asked %s ago — stopping, then waits here" % fmt_dur(now - e["yield_since"])
        else:
            est = "~%s" % fmt_dur(e["est_min"] * 60) if e.get("est_min") else "no estimate"
            if cursor is not None:
                start = fmt_clock(cursor, now)
                cursor = cursor + e["est_min"] * 60 if e.get("est_min") else None
                if cursor is not None:
                    t = "%s, ends ~%s (starts ~%s)" % (est, fmt_clock(cursor, now), start)
                else:
                    t = "%s, starts ~%s, ends: unknown" % (est, start)
            else:
                t = "%s, starts: unknown" % est
            if e.get("skips"):
                t += " (skipped once)"
        out.append((e, t))
    return out


PALETTE = [36, 35, 33, 32, 34, 31]  # per-GPU colors: cyan, magenta, yellow, green, blue, red


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
                i, prio_mark(e), e["owner"][:12], _trim(e["job"], 28), models,
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
                i, prio_mark(e), e["owner"][:12], _trim(e["job"], 28), models, timing))
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
    """Each GPU as a color-ruled card, a blank line between, recent log pulled out below."""
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
    """Each GPU gets a colored left gutter bar, a blank line between."""
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
    """One compact line per GPU: a colored dot, name, mem, state; your card gets an arrow."""
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
        text = "processes on this card: %s" % (", ".join(loaded) or "none over 1 GB")
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


def _bar(frac, n=10):
    if frac is None:
        return "·" * n
    k = max(0, min(n, round(frac * n)))
    return "█" * k + "░" * (n - k)


def summary_lines(views, hosts, width=120):
    """One line per GPU across the fleet, under a header row: GPU, machine (only when there are
    several), memory, utilization, owner and model of what runs. Styled in the machine's color.
    It fits WIDTH by stepping down: 10-cell bars, 5-cell bars, numbers only, trimmed machine
    names; the model is cut to the rest."""
    facts = []
    for host, snap, err in views:
        idx = hosts.index(host) % 6
        if err or snap is None:
            facts.append(dict(name="?", host=host, used=None, gib=None, shared=False, util=None,
                              owner="", what="unreachable", st="m%d" % idx))
            continue
        cards = snap.get("cards") or {}
        card = snap.get("card") or (next(iter(cards)) if len(cards) == 1 else None)
        c = cards.get(card) or {}
        m = (snap.get("meters") or {}).get(card) or {}
        q = snap["entries"]
        r = q[0] if q and q[0]["state"] in ("running", "yielding") else None
        if r:
            owner = ("▶ " if r["state"] == "running" else "⏏ ") + r["owner"]
            what = ",".join(r["models"]) or r["job"]
        else:
            owner, what = "", "idle" if not q else "%d waiting" % len(q)
        facts.append(dict(name=c.get("model") or card or host, host=host, used=m.get("used_mib"),
                          gib=c.get("mib"), shared=bool(c.get("integrated")), util=m.get("util"),
                          owner=owner, what=what, st="m%d" % idx))
    many = len({f["host"] for f in facts}) > 1      # one machine: no machine column
    heads = ["GPU", "HOST", "MEM", "UTIL", "OWNER"]

    def gbs(f):
        """"6.5/24 GB"; an iGPU borrows system RAM, so up to that much: "0.9/≤24 GB"."""
        tot = "%s%d" % ("≤" if f["shared"] else "", round(f["gib"] / 1024)) if f["gib"] else "?"
        return "%s/%s GB" % ("%.1f" % (f["used"] / 1024) if f["used"] is not None else "?", tot)

    def cols(f, bars, host_max):
        used, gib, util = f["used"], f["gib"], f["util"]
        frac = used / gib if used is not None and gib else None
        h = f["host"] if len(f["host"]) <= host_max else f["host"][:host_max - 1] + "…"
        mem = "%s %s" % (_bar(frac, bars), gbs(f)) if bars else gbs(f)
        pct = "%3d%%" % util if util is not None else "  ?"
        gpu = "%s %s" % (_bar(util / 100 if util is not None else None, bars), pct) if bars else pct
        return [f["name"], h if many else "", mem, gpu, _trim(f["owner"], 14)]

    def layout(bars, host_max):
        rows = [cols(f, bars, host_max) for f in facts]
        keep = [i for i in range(5) if any(r[i] for r in rows) or i == 4]
        widths = {i: max([len(r[i]) for r in rows] + [len(heads[i])]) for i in keep}
        fmt = lambda r: "  " + "  ".join(r[i].ljust(widths[i]) for i in keep) + "  "
        return fmt(heads), [fmt(r) for r in rows]

    longest = max((len(f["host"]) for f in facts), default=0)
    # (bar cells, longest machine name): roomiest first, keep ~20 cells for the model
    steps = [(10, longest), (5, longest), (0, longest), (0, 12), (0, 8)]
    for bars, host_max in steps:
        head, lead = layout(bars, host_max)
        if max((len(x) for x in lead), default=0) + 20 <= width:
            break
    out = [((head + "MODEL").rstrip(), "dim")] if facts else []
    for x, f in zip(lead, facts):
        room = max(width - 1 - len(x), 0)
        out.append(((x + _trim(f["what"], room)).rstrip() if room else x.rstrip(),
                    "%s|b2:%d" % (f["st"], 2 + len(f["name"]))))
    return out


IND = " " * 8           # TUI: card info and jobs sit a tab in under the card's rule
DET = IND + " " * 7     # a job's detail lines, under its owner


def tui_lines(views, sel_host, sel_row, now, width=80, only_focus=False):
    """Screen content for the TUI as [(text, style)]. One roomy, color-coded block per GPU,
    showing its full listing: a colored header (the focused host's style ends in "s", and your
    own card is tagged ◀ YOU), what's on the card, then every queued job with its models, timing,
    progress, id (when selected) and why — nothing truncated away. A blank line separates GPUs.
    Styles: run / sel / warn / dim / head / "" / "h<idx>" (+ "s" when focused).
    views = [(host, snap or None, error or None)]. ONLY_FOCUS (the TUI): the summary works as
    tabs, its focused row highlighted, and only the focused GPU's block is shown below it."""
    L, selected = [], None
    hosts = list(dict.fromkeys(v[0] for v in views))
    title = " GPU Queue · %s " % ", ".join(hosts)
    side = max(2, (width - 1 - len(title)) // 2)
    L.append((("─" * side + title + "─" * side)[:max(width - 1, 0)], "title"))
    L.append(("", ""))
    summ = summary_lines(views, hosts, width)
    if only_focus and len(summ) > sel_host + 1:      # row 0 is the column header
        t, st = summ[sel_host + 1]      # the focused GPU: in color, the others dim
        base_st, _, span = st.partition("|")
        summ[sel_host + 1] = (t, base_st + "s" + ("|" + span if span else ""))
    L.extend(summ)
    L.append(("", ""))
    for hi, (host, snap, err) in enumerate(views):
        idx = hosts.index(host) % 6     # one color per machine, its cards share it
        focus = hi == sel_host
        if only_focus and not focus:
            continue
        mark = "" if only_focus else "▸ " if focus else "  "   # which card the keys act on
        hstyle = "h%ds" % idx if focus else "h%d" % idx
        if err or snap is None:
            L.append((mark + (err or host + ": no data"), hstyle))
            L.append(("", ""))
            continue
        q = snap["entries"]
        you = "" if only_focus else "   ◀ YOU" if _is_current(snap) else ""
        base = header(snap, now)
        rule = "─" * max(2, width - len(mark) - len(base) - len(you) - 2)
        line = mark + base + " " + rule + you
        cards = snap.get("cards") or {}
        cname = snap.get("card") or (next(iter(cards)) if len(cards) == 1 else None)
        model = (cards.get(cname) or {}).get("model") or cname or ""
        at = line.find(model) if model else -1
        L.append((line, hstyle + ("|b%d:%d" % (at, at + len(model)) if at >= 0 else "")))
        L.append(("", ""))
        for w in warnings(snap):
            L.append((IND + w, "alert" if w.startswith("⚠") else "dim"))
        if not q:
            L.append((IND + "queue empty", "dim"))
        elif warnings(snap):
            L.append(("", ""))
        for i, (e, timing) in enumerate(entry_timings(q, now)):
            run, yld = e["state"] == "running", e["state"] == "yielding"
            sel = focus and i == sel_row
            if sel:
                selected = (snap, e)
            prim = IND + "%2d %s%s %s" % (i + 1, "▶" if run else "⏏" if yld else " ",
                                    prio_mark(e), e["owner"])
            if e["job"]:
                prim += "  —  " + e["job"]
            L.append((prim, "sel" if sel else "run" if run else "warn" if yld else "wait"))
            det = DET + "%s · %s" % (",".join(e["models"]) or "-", timing)
            if e["state"] in ("running", "yielding"):
                det += " · beat %s ago" % fmt_dur(now - (e["heartbeat"] or e["started"]))
            if e.get("progress"):
                det += " · " + e["progress"]
            L.append((det, "dim"))
            if sel:
                bits = ["id %s" % e["id"], "priority %d" % prio(e), "joined %s" % fmt_clock(e["joined"], now)]
                if e.get("max_min"):
                    bits.append("max %s" % fmt_dur(e["max_min"] * 60))
                L.append((DET + " · ".join(bits), "dim"))
            if e.get("why"):
                L.append((DET + "why: %s" % e["why"], "dim"))
        L.append(("", ""))
        L.append(("", ""))
    foc = views[sel_host] if 0 <= sel_host < len(views) else None
    hist = (foc[1] or {}).get("history") if foc and foc[1] else None
    if hist:
        L.append((" recent ", "head"))
        L.extend((fmt_event(h, now), "dim") for h in hist[-4:])
    return L

# ---- the TUI screen: status first (every GPU, every alert), then one GPU's line ----

def _card_of(snap):
    cards = snap.get("cards") or {}
    card = snap.get("card") or (next(iter(cards)) if len(cards) == 1 else None)
    return card, cards.get(card) or {}


def _runner(q):
    return q[0] if q and q[0]["state"] in ("running", "yielding") else None


def frees_at(snap, now):
    """When the card frees up, for the summary: "now", "~20:07", "soon" (yielding), "over"
    (past its estimate), or "?" (no estimate)."""
    q = snap["entries"]
    r = _runner(q)
    if r is None:
        return "now"
    if r["state"] == "yielding":
        return "soon"
    if not r.get("est_min") or r.get("started") is None:
        return "?"
    end = r["started"] + r["est_min"] * 60
    return "~" + fmt_clock(end, now) if end > now else "over"


def fleet_alerts(views, now):
    """[(view index, text)] — everything that needs a human, on every GPU: warnings, jobs past
    their estimate, held lines, unreachable machines. One line each, named by GPU."""
    out, paused = [], set()
    for vi, (host, snap, err) in enumerate(views):
        if err or snap is None:
            out.append((vi, "%s unreachable" % host))
            continue
        card, c = _card_of(snap)
        name = c.get("model") or card or host
        if snap.get("paused") and host not in paused:
            paused.add(host)
            out.append((vi, "%s: line held — nobody may start" % host))
        r = _runner(snap["entries"])
        if r and r["state"] == "running" and r.get("est_min") and r.get("started") is not None:
            over = now - r["started"] - r["est_min"] * 60
            if over > 0:
                out.append((vi, "%s: %s is %s past its estimate (beat %s ago)" % (
                    name, r["owner"], fmt_dur(over), fmt_dur(now - (r["heartbeat"] or r["started"])))))
        for w in warnings(snap):
            if w.startswith("⚠"):
                out.append((vi, "%s: %s" % (name, w.lstrip("⚠ "))))
    return out


def screen_summary(views, hosts, now, width, focus, flagged=()):
    """The status table: one row per GPU. Columns fit WIDTH by stepping down (bars, then
    numbers; the model column only when there's room; NOW takes what is left)."""
    rows = []
    for hi, (host, snap, err) in enumerate(views):
        idx = hosts.index(host) % 6
        st = "m%d%s" % (idx, "s" if hi == focus else "")
        if err or snap is None:
            rows.append(dict(name="?", host=host, now="unreachable", frees="", next="", used=None,
                             gib=None, shared=False, util=None, model="", st=st))
            continue
        card, c = _card_of(snap)
        m = (snap.get("meters") or {}).get(card) or {}
        q = snap["entries"]
        r = _runner(q)
        waiting = len(q) - (1 if r else 0)
        rows.append(dict(
            name=c.get("model") or card or host, host=host,
            # the runner's priority mark (⚑ ★ ☾) says what a new job would have to beat
            now=("%s%s%s  %s" % ("" if prio_mark(r) == " " else prio_mark(r) + " ", r["owner"],
                                 " (yielding)" if r["state"] == "yielding" else "", r["job"]) if r
                 else "idle" if not q else "free, %d waiting" % waiting),
            frees=frees_at(snap, now), next="+%d" % waiting if r and waiting else "-",
            used=m.get("used_mib"), gib=c.get("mib"), shared=bool(c.get("integrated")),
            util=m.get("util"), model=",".join(r["models"]) if r else "", st=st))
    # the same GPU model on two machines: name the machine too
    names = [f["name"] for f in rows]
    for f in rows:
        if names.count(f["name"]) > 1:
            f["name"] += " @" + f["host"]

    def mem(f, bars):
        tot = ("%d%s" % (round(f["gib"] / 1024), "G sh" if f["shared"] else "G")) if f["gib"] else "?"
        txt = "%s/%s" % ("%.0f" % (f["used"] / 1024) if f["used"] is not None else "?", tot)
        frac = f["used"] / f["gib"] if f["used"] is not None and f["gib"] else None
        return "%s %s" % (_bar(frac, bars), txt) if bars else txt

    def util(f, bars):
        pct = "%3d%%" % f["util"] if f["util"] is not None else "  ?"
        return "%s %s" % (_bar(f["util"] / 100 if f["util"] is not None else None, bars), pct) if bars else pct

    def build(bars, show_model):
        heads = ["GPU", "NOW", "FREES", "NEXT", "MEM", "UTIL"] + (["MODEL"] if show_model else [])
        table = [[f["name"], f["now"], f["frees"], f["next"], mem(f, bars), util(f, bars)] +
                 ([f["model"]] if show_model else []) for f in rows]
        ni = heads.index("NOW")
        widths = [max([len(r[i]) for r in table] + [len(heads[i])]) for i in range(len(heads))]
        fixed = sum(w for i, w in enumerate(widths) if i != ni) + 2 * (len(heads) - 1) + 2
        widths[ni] = max(10, min(widths[ni], width - 1 - fixed))
        fmt = lambda r: "  " + "  ".join(_trim(x, widths[i]).ljust(widths[i]) for i, x in enumerate(r))
        return fixed + widths[ni], heads, table, fmt, widths

    # (bar cells, model column, the least room NOW may get): roomiest first
    for bars, show_model, now_min in ((10, True, 36), (10, False, 28), (5, False, 24), (0, False, 0)):
        total, heads, table, fmt, widths = build(bars, show_model)
        want = min(now_min, max(len(r[heads.index("NOW")]) for r in table))
        if total <= width - 1 and widths[heads.index("NOW")] >= want:
            break
    out = [(fmt(heads).rstrip(), "dim")]
    col = {hd: 2 + sum(widths[:i]) + 2 * i for i, hd in enumerate(heads)}
    for vi, (r, f) in enumerate(zip(table, rows)):
        line = fmt(r).rstrip()
        # color, sparingly: red ⚠ = needs you, yellow = watch it, green = free now; the
        # focused GPU is the one whose name is bold (the rule below repeats it)
        spans = []
        if vi in flagged:               # ⚠ in the margin: this GPU has an alert below
            line = "⚠" + line[1:]
            spans.append("r0:1")
        if f["st"].endswith("s"):
            spans.append("b2:%d" % (2 + len(f["name"])))
        fr = col["FREES"]
        if f["frees"] == "now":
            spans.append("g%d:%d" % (fr, fr + 3))
        elif f["frees"] in ("over", "soon"):
            spans.append("y%d:%d" % (fr, fr + 4))
        if f["used"] is not None and f["gib"] and f["used"] / f["gib"] >= 0.95:
            spans.append("y%d:%d" % (col["MEM"], col["MEM"] + widths[heads.index("MEM")]))
        out.append((line[:max(width - 1, 0)], f["st"] + "".join("|" + x for x in spans)))
    return out


def _blocked(snap, q, i, now):
    """Why a waiting job isn't running yet, in one plain line."""
    if snap.get("paused"):
        return "blocked: the queue is paused"
    y = yielding(q)
    if y:
        return "blocked: %s (%s) is still stopping to yield" % (y[0]["owner"], y[0]["id"])
    if i == 0:
        return "next: it starts as soon as it calls `gpuq start` (within 15 min, or it is skipped)"
    return "blocked: %d ahead · %s has the card, frees %s" % (i, q[0]["owner"], frees_at(snap, now))


def tui_screen(views, sel_host, sel_row, now, width=80):
    """[(text, style)] for the TUI: a title bar, the status table for every GPU, alerts for every
    GPU, then the focused GPU's line (two rows per job) with a detail block for the selected job."""
    hosts = list(dict.fromkeys(v[0] for v in views))
    alerts = fleet_alerts(views, now)
    title = " GPU Queue · %d GPU%s%s " % (len(views), "s" * (len(views) != 1),
                                          " · ⚠ %d need%s you" % (len(alerts), "s" * (len(alerts) == 1))
                                          if alerts else "")
    L = [(_rule(title, width), "title")]
    L.extend(screen_summary(views, hosts, now, width, sel_host, {vi for vi, _ in alerts}))
    L.extend(("  ⚠ " + _trim(a, width - 6), "alert") for _, a in alerts)
    if not (0 <= sel_host < len(views)):
        return L
    host, snap, err = views[sel_host]
    idx = hosts.index(host) % 6
    if err or snap is None:
        L.append((_rule(" %s " % host, width), "h%d" % idx))
        L.append(("  " + (err or "no data"), "alert"))
        return L
    card, c = _card_of(snap)
    name = c.get("model") or card or host
    label = " %s · %s " % (name, host)
    rule = _rule(label, width)
    at = rule.find(name)
    L.append((rule, "h%d|b%d:%d" % (idx, at, at + len(name))))
    q = snap["entries"]
    if not q:
        L.append(("  queue empty: free now", "dim"))
    sel = None
    for i, (e, timing) in enumerate(entry_timings(q, now)):
        run, yld = e["state"] == "running", e["state"] == "yielding"
        line = "  %d %s%s %-12s  %s" % (i + 1, "▶" if run else "⏏" if yld else " ", prio_mark(e),
                                        e["owner"][:12], e["job"])
        is_sel = i == sel_row
        if is_sel:
            sel = (i, e)
        L.append((_trim(line, width - 1), "sel" if is_sel else "warn" if yld else "wait"))
        bits = [timing]
        if run or yld:
            bits.append("beat %s ago" % fmt_dur(now - (e["heartbeat"] or e["started"])))
        if e["models"]:
            bits.append(",".join(e["models"]))
        if e.get("progress"):
            bits.append(e["progress"])
        L.append((_trim("          " + " · ".join(bits), width - 1), "dim"))
    if sel:
        i, e = sel
        bits = ["#%d %s" % (i + 1, e["id"]), "priority %d" % prio(e), "joined %s" % fmt_clock(e["joined"], now)]
        if e.get("max_min"):
            bits.append("max %s" % fmt_dur(e["max_min"] * 60))
        L.append(("", ""))
        L.append((_trim(" ┄ " + " · ".join(bits), width - 1), "dim"))
        if e["state"] == "waiting":
            L.append((_trim("   " + _blocked(snap, q, i, now), width - 1), "wait"))
        if e.get("why"):
            L.append((_trim("   why: " + e["why"], width - 1), "dim"))
    info = [w for w in warnings(snap) if not w.startswith("⚠")]
    for w in info:
        L.append((_trim("   " + w, width - 1), "dim"))
    hist = snap.get("history") or []
    if hist:
        L.append((_trim(" recent: " + " · ".join(fmt_event(x, now) for x in hist[-2:][::-1]), width - 1), "dim"))
    return L


RULE_INDENT = 12   # rule labels start here and stay put; the line runs on to the edge


def _rule(label, width):
    return ("─" * RULE_INDENT + label + "─" * max(width, 0))[:max(width - 1, 0)]


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


def nvidia_model(full):
    """"NVIDIA GeForce RTX 3090" -> "RTX 3090"."""
    words = [w for w in full.split() if w not in ("NVIDIA", "GeForce")]
    return " ".join(words) or full


def amd_model(full):
    """"Strix [Radeon 880M / 890M]" -> "Radeon 880M": the first name inside the brackets."""
    inner = full[full.find("[") + 1:full.rfind("]")] if "[" in full else full
    return inner.split("/")[0].strip() or full


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
            cards[name] = {"uuid": uuid, "mib": int(float(mib)), "model": nvidia_model(model)}
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
        # A host that had no detectable card kept its line under the placeholder name "gpu".
        # Once a card is detected, that line moves to the card's (empty) line, running job and all.
        if cards and "gpu" in found and "gpu" not in cards:
            legacy = found.pop("gpu")
            if legacy and not found.get(default):
                for e in legacy:
                    e["gpu"] = default
                found[default] = legacy
            elif legacy:
                found["gpu"] = legacy      # the card's line is in use: leave both alone
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


def read_freed():
    """{unit: card} for resident units `gpuq free` stopped (restore starts them again)."""
    try:
        return json.loads((STATE / "freed.json").read_text()).get("units") or {}
    except (OSError, ValueError, AttributeError):
        return {}


def write_freed(units):
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / "freed.json"
    if units:
        path.write_text(json.dumps({"units": units}))
    elif path.exists():
        path.unlink()


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


def history_all():
    """Every history record (for estimates). Bad lines are skipped."""
    try:
        text = (STATE / "history.jsonl").read_text()
    except FileNotFoundError:
        return []
    out = []
    for l in text.splitlines():
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


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
        out[name] = {"vendor": "amd", "model": amd_model(_pci_name(dev)),
                     "uuid": uid and "GPU-" + uid.zfill(16) or pci or c.name,
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


def card_meters(cards):
    """{card: {"used_mib", "util"}}: memory in use and utilization percent per card, for the TUI's
    summary bars. nvidia-smi for NVIDIA; sysfs for AMD (an iGPU's memory is its GTT use).
    Either value may be None. Empty under GPUQ_CARDS (the tests' fake cards)."""
    out = {}
    if os.environ.get("GPUQ_CARDS"):
        return out
    by_uuid = {c["uuid"]: n for n, c in cards.items() if c.get("vendor") != "amd" and c.get("uuid")}
    if by_uuid:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            for line in r.stdout.splitlines() if r.returncode == 0 else []:
                uuid, used, util = [x.strip() for x in line.split(",")]
                if uuid in by_uuid:
                    out[by_uuid[uuid]] = {"used_mib": int(used) if used.isdigit() else None,
                                          "util": int(util) if util.isdigit() else None}
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    for n, c in cards.items():
        if c.get("vendor") == "amd":
            d = SYSFS / "class" / "drm" / _card_dir(c) / "device"
            used = _read(d / ("mem_info_gtt_used" if c.get("integrated") else "mem_info_vram_used"))
            util = _read(d / "gpu_busy_percent")
            out[n] = {"used_mib": int(used) >> 20 if used.isdigit() else None,
                      "util": int(util) if util.isdigit() else None}
    return out


def local_snapshot(now):
    with store(write=False) as Q:
        lines, cards = Q["lines"], Q["cards"]
        use = card_usage(cards, fleet.resident_names(config()))
        has_amd = any(c.get("vendor") == "amd" for c in cards.values())
        if use is None or (len(cards) < 2 and not has_amd):
            loaded = gpu_loaded()
        else:
            loaded = [j for u in use.values() for j in u["jobs"]]
        snap = snapshot(lines[Q["default"]], now, history(), loaded, paused=paused_state(),
                        lines=lines, cards=cards, card_use=use, default=Q["default"], evac=read_evac())
        snap["meters"] = card_meters(cards)
        if use and len(use) == 1:       # one card: per_card() leaves the snapshot alone, so carry busy here
            snap["busy"] = next(iter(use.values())).get("busy")
        return snap


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


TUI_KEYS = " ↑↓ job  ←→ GPU  k/j move  d remove  p hold  ? help  q quit "
TUI_HELP = """GPU Queue — help                                         any key closes

KEYS
  ↑ ↓          select a job in the GPU's line below
  ← → / Tab    switch GPU (the GPU with the bold name is the one shown)
  k / Shift-↑  move the selected job up, sooner;  j / Shift-↓  down, later.
               Moving the running job down asks it to yield: it stops at its next
               heartbeat, waits, and resumes when it is #1 again.
  d            remove the selected job from the line (its process is not killed)
  p            hold / release the machine's line: while held nobody may start,
               on any of that machine's GPUs. Running jobs keep running.
  r            refresh now (it refreshes every few seconds anyway)
  q / Esc      quit; jobs are not affected

SUMMARY
  NOW          who holds the card and what they run; idle, or free with jobs waiting
  FREES        when the card frees up: a time, now, soon (yielding), over (past its
               estimate, end unknown), ? (no estimate)
  NEXT         how many jobs wait behind it
  MEM / UTIL   memory in use / total (sh = an iGPU's shared system memory) and load
  ⚠            something needs you, on any GPU: a job past its estimate, one that
               won't yield, GPU use outside the queue, a held line, a lost machine

MARKS
  ▶ running   ⏏ yielding   ⚑ ablaze (30)   ★ important (20)   ☾ chill (5)
  blank = normal (10). Higher priority goes first; outranking the running job
  makes it yield. Short jobs may pass normal and chill jobs, never important ones.

COLORS  (always next to a word or symbol that says the same)
  red ⚠      needs you
  yellow     temporary, watch it: yielding, over its estimate, memory nearly full
  green now  free now: a job could start on that GPU
  bold name  the GPU whose line is shown below
  dim        detail""".splitlines()
HELP_COLORS = [("red ⚠", "r"), ("yellow", "y"), ("green now", "g"), ("bold name", "b"), ("⚠", "r"),
               ("⚑", "b"), ("over", "y"), ("soon", "y"), ("⏏", "y")]


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
        # Three colors, one meaning each, always beside a word or symbol that says the same:
        # red + ⚠ = needs you, yellow (bold, for light themes) = temporary, watch it,
        # green = free now. NO_COLOR or a colorless terminal: bold/reverse only.
        colors = curses.has_colors() and not os.environ.get("NO_COLOR")
        if colors:
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
        green = curses.color_pair(1) if colors else curses.A_BOLD
        yellow = (curses.color_pair(2) if colors else 0) | curses.A_BOLD
        red = (curses.color_pair(3) if colors else 0) | curses.A_BOLD
        span_attr = {"b": curses.A_BOLD, "g": green, "y": yellow, "r": red}
        # bold: the title and the focused GPU's name; reverse: the title, the selected job, the key bar
        style = {"title": curses.A_BOLD | curses.A_REVERSE, "head": curses.A_UNDERLINE,
                 "run": curses.A_NORMAL, "sel": curses.A_REVERSE,
                 "warn": yellow, "alert": red,
                 "wait": curses.A_NORMAL, "dim": curses.A_DIM, "dimhead": curses.A_DIM,
                 "": curses.A_NORMAL}

        def resolve(st):
            """A fixed style name, or a GPU-colored header key 'h<idx>' (+ 's' when focused)."""
            if len(st) >= 2 and st[0] in "hm" and st[1].isdigit():
                return curses.A_NORMAL      # summary rows and the rule: plain, never dimmed
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

        def put(y, text, attr, spans=()):
            """Draw a line, then overlay its spans: (kind, a, b) with kind b/g/y/r."""
            try:
                scr.addnstr(y, 0, text.ljust(w), w - 1, attr)
                for kind, a, b in spans:
                    if a < w - 1:
                        scr.addnstr(y, a, text[a:b], max(min(b, w - 1) - a, 0), attr | span_attr[kind])
            except curses.error:
                pass

        def parse(st):
            st, *rest = st.split("|")
            spans = []
            for x in rest:
                a, _, b = x[1:].partition(":")
                if x[:1] in span_attr and a.isdigit() and b.isdigit():
                    spans.append((x[0], int(a), int(b)))
            return st, spans

        while True:
            if time.time() - fetched >= args.every:
                views, fetched = fetch_views(hosts), time.time()
            sr = max(0, min(sr, len(entries()) - 1))
            h, w = scr.getmaxyx()
            scr.erase()
            for y, (text, st) in enumerate(tui_screen(views, sh, sr, time.time(), w)[:h - 1]):
                st, spans = parse(st)
                put(y, text, resolve(st), spans)
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
            elif k == ord("?"):
                scr.erase()
                for y, text in enumerate(TUI_HELP[:h]):
                    line = " " + text
                    spans = [(k, line.find(word), line.find(word) + len(word)) for word, k in HELP_COLORS
                             if word in line]
                    put(y, line, curses.A_REVERSE | curses.A_BOLD if y == 0 else
                        curses.A_BOLD if text.isupper() else curses.A_NORMAL, spans)
                scr.refresh()
                scr.timeout(-1)
                scr.getch()
                scr.timeout(250)
            elif k == ord("r"):
                fetched = 0
            elif k == ord("p") and views[sh][1]:
                verb = "resume" if views[sh][1].get("paused") else "pause"
                code, out = act(host, verb)
                msg = " " + (out or "%s: line %s" % (host, "held — nobody may start" if verb == "pause"
                                                       else "released")) + " "
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


def apply_rules(cfg):
    """Config "rules" tune the line: {"jump_ratio": 10, "jump_max_hours": 6,
    "priorities": {"chill": 5, "normal": 10, "important": 20, "ablaze": 30}}."""
    from . import core
    r = cfg.get("rules") or {}
    if r.get("jump_ratio") is not None:
        core.JUMP_RATIO = float(r["jump_ratio"])
    if r.get("jump_max_hours") is not None:
        core.JUMP_MAX_SEC = float(r["jump_max_hours"]) * 3600
    for k, v in (r.get("priorities") or {}).items():
        if k in core.PRIORITY:
            core.PRIORITY[k] = int(v)
    core.SHIELD = core.PRIORITY["important"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    apply_rules(config())
    ap = argparse.ArgumentParser(prog="gpuq", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", help="run the command against HOST's queue over ssh")
    sub = ap.add_subparsers(dest="cmd")
    card = argparse.ArgumentParser(add_help=False)
    card.add_argument("--gpu", help="which card's line, by model number (e.g. 3060); see `gpuq show`")
    view = argparse.ArgumentParser(add_help=False)   # layout flag, shared by show/watch/tui
    view.add_argument("--mode", choices=["classic", "cards", "bars", "list"], default="classic",
                      help="layout: classic (default), cards, bars, or list")
    jsonp = argparse.ArgumentParser(add_help=False)  # --json on action commands: one JSON object out
    jsonp.add_argument("--json", action="store_true",
                       help="print one JSON object: ok, code, state, message, plus the command's fields")
    need = argparse.ArgumentParser(add_help=False)   # what a job needs from a card
    need.add_argument("--vram", type=float, metavar="GB", help="GPU memory the job needs, in GB")
    need.add_argument("--tag", action="append", default=None, metavar="TAG",
                      help="card tag the job needs (cuda, rocm, igpu, or your own); repeatable")
    prio_p = argparse.ArgumentParser(add_help=False)  # where a job goes in line
    g = prio_p.add_mutually_exclusive_group()
    g.add_argument("--priority", metavar="LEVEL|N",
                   help="%s, preempt (one above the running job), or any integer; higher goes first,"
                        " and outranking the running job makes it yield"
                        % ", ".join("%s (%d)" % kv for kv in PRIORITY.items()))
    for lvl, n in PRIORITY.items():
        g.add_argument("--" + lvl, dest="level", action="store_const", const=lvl,
                       help="priority %d%s" % (n, " (never jumped by short jobs)" if n >= 20 else ""))
    g.add_argument("--preempt", dest="level", action="store_const", const="preempt",
                   help="one above the running job's priority: it yields, you go next")
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
    p = sub.add_parser("join", help="get in line; prints your id", parents=[card, need, prio_p, jsonp])
    p.add_argument("--spec", metavar="FILE",
                   help="read the job as a JSON object from FILE (- = stdin): owner, job, why, models,"
                        " est, max, priority (or level: chill/normal/important/ablaze/preempt), gpu,"
                        " vram, tags; flags given as well win")
    p.add_argument("--owner")
    p.add_argument("--job",
                   help="what the job is — be descriptive; the tui shows it in full, short views trim")
    p.add_argument("--why", default="",
                   help="why it's running / who it's for — a full phrase is fine; shown in full in the tui")
    p.add_argument("--models", default="", help="comma-separated")
    p.add_argument("--est", type=int, help="expected minutes (default: the median of this job's past runs)")
    p.add_argument("--max", type=int, help="minutes after which the sweep purges you")
    p.add_argument("--no-history", action="store_true", help="don't estimate from past runs")
    p = sub.add_parser("edit", help="change an entry's details; id and place are kept"
                       " (--gpu moves a waiting entry to the end of that card's line)",
                       parents=[card, need, prio_p, jsonp])
    p.add_argument("id")
    for f in ("--job", "--why", "--models"):
        p.add_argument(f)
    p.add_argument("--est", type=int)
    p.add_argument("--max", type=int)
    p = sub.add_parser("start", help="take the GPU (only as #1)", parents=[jsonp])
    p.add_argument("id")
    p.add_argument("--adopt", action="store_true",
                   help="take over a lock file written outside gpuq (backfilling a live job)")
    p = sub.add_parser("beat", help="heartbeat; at least every 15 min (exit 5 = yield: stop, poll start)",
                       parents=[jsonp])
    p.add_argument("id")
    p.add_argument("--progress")
    p = sub.add_parser("done", help="release the GPU and leave the line", parents=[jsonp])
    p.add_argument("id")
    p = sub.add_parser("env", help="print the job's card as CUDA_VISIBLE_DEVICES=GPU-<uuid>", parents=[jsonp])
    p.add_argument("id")
    p = sub.add_parser("mv", help="move an entry (position or id) to a position", parents=[card, jsonp])
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
    p.add_argument("--free", action="store_true",
                   help="free the card's residents (`gpuq free`) once granted; restart what it stopped at the end")
    p = sub.add_parser("yield", help="ask the running job to step back (default to #2)", parents=[jsonp])
    p.add_argument("id")
    p.add_argument("to", nargs="?")
    p = sub.add_parser("rm", help="remove an entry (position or id)", parents=[card, jsonp])
    p.add_argument("ref")
    sub.add_parser("pause", help="hold the line: nobody may start until resume")
    sub.add_parser("resume", help="release the line; #1 gets a fresh 15 minutes")
    sub.add_parser("sweep", help="garbage collection (the timer runs this)")
    p = sub.add_parser("evac", help="move a card's resident services off it, to free the whole card")
    p.add_argument("card", nargs="?", help="card to empty (default: the config default card)")
    p.add_argument("--to", help="card to move residents to (default: the other card)")
    p.add_argument("--dry-run", action="store_true", help="show what would move; change nothing")
    p = sub.add_parser("restore", help="undo an evac or a free: residents back on their home card, running")
    p.add_argument("card", nargs="?")
    p.add_argument("--dry-run", action="store_true", help="show what would change; change nothing")
    p = sub.add_parser("free", help="release a card's resident memory: unload models, or stop units")
    p.add_argument("card", nargs="?", help="card (default: the host default)")
    p.add_argument("--dry-run", action="store_true", help="show what would be done; change nothing")
    p = sub.add_parser("inventory", help="what every card offers: usable memory, tags, residents, wait")
    p.add_argument("--json", action="store_true")
    p.add_argument("--hosts", default=hosts_default, help="comma-separated (default: %s)" % hosts_default)
    p.add_argument("--local", action="store_true", help="this host only")
    p = sub.add_parser("where", help="which card, on which host, can take a job — and how soon",
                       parents=[need, prio_p])
    p.add_argument("--est", type=int, help="the job's minutes (applies the short-job jump to the wait)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--hosts", default=hosts_default, help="comma-separated (default: %s)" % hosts_default)
    p.add_argument("--local", action="store_true", help="this host only")
    p = sub.add_parser("init", help="write a starter config from the cards (and servers) found here")
    p.add_argument("--print", action="store_true", help="print it; write nothing")
    p.add_argument("--force", action="store_true", help="replace an existing config (kept as .bak)")
    sub.add_parser("mcp", help="serve the queue to AI agents over MCP (stdio)")
    # everything after the first "--" is gpuq run's command, never gpuq's own options
    head, run_cmd = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    args = ap.parse_args(head)
    args.run_cmd = run_cmd

    if args.host and args.host != HOST:
        return _forward(argv, args.host)

    now = float(os.environ.get("GPUQ_NOW") or time.time())
    as_json = getattr(args, "json", False) and args.cmd in ACTION_JSON
    args.result = {}
    try:
        rc = _run(args, now)
    except QueueError as err:
        if as_json:
            print(json.dumps({"ok": False, "code": err.code, "state": state_name(args.cmd, err.code),
                              "message": str(err)}))
        else:
            print("gpuq: %s" % err, file=sys.stderr)
        return err.code
    if as_json:
        print(json.dumps(dict({"ok": rc == 0, "code": rc, "state": state_name(args.cmd, rc)},
                              **args.result)))
    return rc


ACTION_JSON = {"join", "edit", "start", "beat", "done", "env", "mv", "yield", "rm"}
STATES = {"start": {0: "go", 3: "wait", 4: "gone"}, "beat": {0: "ok", 4: "purged", 5: "yield"}}


def state_name(cmd, code):
    """The word for an exit code, as --json and MCP report it."""
    return STATES.get(cmd, {}).get(code) or ("ok" if code == 0 else "error")


def _run(args, now):
    cmd = args.cmd
    if cmd is None:
        # bare `gpuq` at a terminal opens the TUI, like htop; piped or scripted, it prints the queue
        if sys.stdin.isatty() and sys.stdout.isatty() and not args.host:
            args.hosts = config().get("hosts") or HOST
            args.every = 5
            return _tui(args)
        cmd = "show"
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
    if cmd == "free":
        return _free(args, now)
    if cmd == "inventory":
        return _inventory(args, now)
    if cmd == "where":
        return _where(args, now)
    if cmd == "init":
        return _init(args, now)
    if cmd == "mcp":
        from .mcp import serve
        return serve()
    res = getattr(args, "result", {})
    as_json = getattr(args, "json", False)
    if cmd == "join":
        _load_spec(args)
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
            models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
            cfg = config()
            est, est_from = args.est, "given" if args.est else None
            if est is None and not args.no_history:
                got = fleet.estimate(history_all(), args.job, args.owner)
                if got:
                    est, est_from = got[0], "history: median of %d run%s" % (got[1], "s" * (got[1] != 1))
                    print("estimate from history: ~%s (%s)" % (fmt_dur(est * 60), est_from), file=sys.stderr)
            tags = args.tag or []
            prefer = cfg.get("owners", {}).get(args.owner)
            if args.gpu:
                name = args.gpu
                line_named(name)
                if args.vram or tags:
                    inv = _inv_local(Q, cfg, now)
                    ok, why = fleet.fits(inv[name], args.vram, tags)
                    if not ok:
                        raise QueueError(2, "the %s can't take it (%s) — see `gpuq where%s`"
                                         % (name, why, _need_flags(args)))
            elif args.vram or tags:
                inv = _inv_local(Q, cfg, now)
                rows = fleet.rank(inv, lines, now, args.vram, tags, est, prefer=prefer,
                                  priority=want_priority(args, None))
                if not rows or not rows[0][1]:
                    raise QueueError(2, "no card here fits (%s) — `gpuq where%s` looks on every host"
                                     % ("; ".join("%s: %s" % (r[0], r[2]) for r in rows) or "no cards",
                                        _need_flags(args)))
                name = rows[0][0]
            else:
                name = prefer or Q["default"]
            q = line_named(name)
            taken = {x["id"] for other in lines.values() for x in other}
            before = {x["id"]: x["state"] for x in q}
            pr = want_priority(args, q)
            e = join(q, now, args.owner, args.job, args.why or "", models, est, args.max,
                     taken=taken, priority=PRIORITY["normal"] if pr is None else pr)
            e["gpu"] = name
            if args.vram:
                e["vram_gb"] = args.vram
            if tags:
                e["tags"] = tags
            log(now, "joined", e, "#%d%s%s" % (q.index(e) + 1, where(q),
                                               "" if prio(e) == PRIORITY["normal"] else ", priority %d" % prio(e)))
            _note_yields(q, now, before)
            jumped = _jump(q, now, e)
            res.update(id=e["id"], gpu=name, position=q.index(e) + 1, est_min=est, priority=prio(e),
                       est_from=est_from, jumped=jumped)
            if not as_json:
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
            pin = card_env(Q, args.id)
            res.update(id=e["id"], gpu=e.get("gpu") or Q["default"],
                       env=dict([pin.split("=", 1)]) if pin else {})
            if multi and pin and not as_json:
                print(pin)
        elif cmd == "env":
            line_of(args.id, by_id=True)
            pin = card_env(Q, args.id)
            res["env"] = dict([pin.split("=", 1)]) if pin else {}
            if not as_json:
                print(pin or "")
        elif cmd == "edit":
            q = line_of(args.id)
            e = q[find(q, args.id)]
            for key, val in (("job", args.job), ("why", args.why), ("est_min", args.est),
                             ("max_min", args.max)):
                if val is not None:
                    e[key] = val
            if args.models is not None:
                e["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
            pr = want_priority(args, q, e)
            if pr is not None:
                e["priority"], e["important"] = pr, pr >= PRIORITY["important"]
            if args.vram is not None:
                e["vram_gb"] = args.vram
            if args.tag is not None:
                e["tags"] = args.tag
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
            if pr is not None or args.gpu:
                before = {x["id"]: x["state"] for x in q}
                place(q, now, e["id"])
                _note_yields(q, now, before)
            log(now, "edited", e, where(q).strip())
            if args.est is not None or pr is not None or args.gpu:
                _jump(q, now, e)
            res.update(id=e["id"], gpu=e.get("gpu"), position=q.index(e) + 1, priority=prio(e))
        elif cmd == "beat":
            q = line_of(args.id, by_id=True)
            res["id"] = args.id
            e = beat(q, now, args.id, args.progress)
            res["position"] = q.index(e) + 1
        elif cmd == "done":
            e = done(line_of(args.id, by_id=True), now, args.id)
            clear_compat(e["id"])
            log(now, "done", e)
            res["id"] = e["id"]
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
    freed = []      # units this run stopped with --free: started again when it ends

    def unfree():
        if freed:
            say("restarting what --free stopped: %s" % ", ".join(freed))
            _restart_all(freed, "start")
            left = read_freed()
            write_freed({u: c for u, c in left.items() if u not in freed})
            freed.clear()

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
                mine = next((n for n, q in Q["lines"].items() if any(e["id"] == args.id for e in q)), None)
            if args.free and mine:
                _, stopped = free_card(mine, config(), say=lambda t: say(t.strip()))
                freed.extend(u for u in stopped if u not in freed)
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
                unfree()
                return code
            stop(proc)
            proc = None
            if rc == 4:
                say("purged from the queue — stopped")
                unfree()
                return 4
            say("yielded — stopped, waiting for the GPU again")
    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            stop(proc)
        gq("done", args.id)
        say("interrupted — stopped and left the line")
        unfree()
        return 130


def _load_spec(args):
    """join --spec FILE|-: fill the flags from a JSON object; a flag given on the command line wins."""
    if args.spec:
        try:
            spec = json.load(sys.stdin if args.spec == "-" else open(args.spec))
        except (OSError, ValueError) as err:
            raise QueueError(2, "--spec: %s" % err)
        if not isinstance(spec, dict):
            raise QueueError(2, "--spec: expected a JSON object")
        keys = {"owner": "owner", "job": "job", "why": "why", "models": "models", "est": "est",
                "est_min": "est", "max": "max", "max_min": "max", "priority": "priority", "level": "level",
                "gpu": "gpu", "vram": "vram", "vram_gb": "vram", "tags": "tag", "tag": "tag"}
        for k, v in spec.items():
            if k == "important":            # the old boolean
                if v and not args.level and args.priority is None:
                    args.level = "important"
                continue
            if k not in keys:
                raise QueueError(2, "--spec: unknown key %r (known: %s)" % (k, ", ".join(sorted(keys))))
            dest = keys[k]
            if dest == "models" and isinstance(v, list):
                v = ",".join(v)
            if dest == "tag" and isinstance(v, str):
                v = [v]
            if getattr(args, dest, None) in (None, "", False):
                setattr(args, dest, v)
    if not args.owner or not args.job:
        raise QueueError(2, "join needs --owner and --job (or a --spec with them)")
    if args.level and args.level not in PRIORITY and args.level != "preempt":
        raise QueueError(2, "--spec: level must be one of %s, preempt" % ", ".join(PRIORITY))


def _need_flags(args):
    return "".join([" --vram %g" % args.vram] if args.vram else []) + \
        "".join(" --tag %s" % t for t in args.tag or [])


def _inv_local(Q, cfg, now, use=None):
    return fleet.inventory(Q["cards"], cfg, Q["lines"], now, use, read_evac(), read_freed())


def local_inventory(now):
    cfg = config()
    with store(write=False) as Q:
        use = card_usage(Q["cards"], fleet.resident_names(cfg))
        return _inv_local(Q, cfg, now, use)


def _remote_json(host, *args):
    """(parsed JSON, None) or (None, error) from `gpuq ARGS` on HOST over ssh."""
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                            remote_cmd() + " " + shlex.join(args)],
                           capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None, "%s: no answer within 30 s" % host
    try:
        return json.loads(r.stdout), None
    except ValueError:
        return None, "%s: %s" % (host, (r.stderr.strip().splitlines() or ["unreachable"])[-1])


def _hosts(args):
    if args.local:
        return [HOST]
    return [h.strip() for h in (args.hosts or HOST).split(",") if h.strip()] or [HOST]


_gb = fleet.gb


def _wait_text(sec):
    return "unknown" if sec is None else "now" if sec < 60 else "~" + fmt_dur(sec)


def _inventory(args, now):
    out = {}
    for h in _hosts(args):
        if h == HOST:
            out[h] = local_inventory(now)
        else:
            got, err = _remote_json(h, "inventory", "--json", "--local")
            out[h] = got if got is not None else {"error": err}
    if args.json:
        print(json.dumps(out if not args.local else out[HOST]))
        return 0
    for h, inv in out.items():
        print(h)
        if "error" in inv and isinstance(inv["error"], str):
            print("  " + inv["error"])
            continue
        for n, c in inv.items():
            bits = []
            if c["reserve_mib"]:
                bits.append("reserve %s" % _gb(c["reserve_mib"]))
            if c["resident_mib"]:
                bits.append("residents %s" % _gb(c["resident_mib"]))
            if c["cap_mib"] != c["mib"]:
                bits.append("capped from %s" % _gb(c["mib"]))
            size = _gb(c["mib"]) + (" shared" if c["integrated"] else "")
            print("  %-6s %-14s usable %-6s %s" % (n, size, _gb(c["usable_mib"]),
                                                   "(%s)" % ", ".join(bits) if bits else ""))
            line = "%d in line, wait %s" % (c["in_line"], _wait_text(c["wait_sec"]))
            if c["running"]:
                line += " · running: %s (%s)" % (c["running"]["job"], c["running"]["owner"])
            print("         tags %s · %s" % (",".join(c["tags"]) or "-", line))
            if c["residents"]:
                print("         residents: %s" % ", ".join(c["residents"]))
    return 0


def where_rows(inv, lines, now, vram=None, tags=(), est=None, priority=None):
    rows = []
    for n, ok, why, wait in fleet.rank(inv, lines, now, vram, tags, est, priority=priority):
        c = inv[n]
        rows.append({"card": n, "fits": ok, "reason": why, "wait_sec": wait,
                     "usable_mib": c["usable_mib"], "tags": c["tags"], "in_line": c["in_line"]})
    return rows


def _where(args, now):
    rows = []
    for h in _hosts(args):
        if h == HOST:
            cfg = config()
            with store(write=False) as Q:
                got = where_rows(_inv_local(Q, cfg, now), Q["lines"], now, args.vram,
                                 args.tag or [], args.est, want_priority(args, None))
        else:
            extra = ([] if args.vram is None else ["--vram", "%g" % args.vram]) + \
                [x for t in args.tag or [] for x in ("--tag", t)] + \
                ([] if args.est is None else ["--est", str(args.est)]) + \
                (["--priority", str(args.priority)] if args.priority is not None else []) + \
                (["--" + args.level] if args.level else [])
            got, err = _remote_json(h, "where", "--json", "--local", *extra)
            if got is None:
                rows.append({"host": h, "card": None, "fits": False, "reason": err, "wait_sec": None,
                             "usable_mib": None, "tags": [], "in_line": 0})
                continue
        rows.extend(dict(r, host=h) for r in got)
    big = float("inf")
    rows.sort(key=lambda r: (not r["fits"], big if r["wait_sec"] is None else r["wait_sec"],
                             r["host"] != HOST, r["usable_mib"] or 0))
    if args.json:
        print(json.dumps([{k: v for k, v in r.items() if not (args.local and k == "host")} for r in rows]))
        return 0 if any(r["fits"] for r in rows) else 1
    for r in rows:
        if r["card"] is None:
            print("✗ %s" % r["reason"])
            continue
        print("%s %-10s %-6s usable %-6s wait %-8s %s" % (
            "✓" if r["fits"] else "✗", r["host"], r["card"], _gb(r["usable_mib"]),
            _wait_text(r["wait_sec"]) if r["fits"] else "-", r["reason"] or ",".join(r["tags"])))
    best = next((r for r in rows if r["fits"]), None)
    if best:
        print("→ gpuq%s join --gpu %s%s ..." % ("" if best["host"] == HOST else " --host " + best["host"],
                                               best["card"], _need_flags(args)))
        return 0
    print("nothing fits%s" % ("" if len(_hosts(args)) > 1 else " here — `--hosts a,b` looks further"))
    return 1


def _init(args, now):
    cfg = config()
    cards = gpu_cards()
    use = card_usage(cards, {}) or {}
    new = dict(cfg)
    new.setdefault("default", default_card(cards))
    new.setdefault("hosts", HOST)
    cc = new.setdefault("cards", {})
    for n, c in cards.items():
        cc.setdefault(n, {"reserve_gb": 0, "tags": []})
        if c.get("integrated"):
            cc[n].setdefault("vram_gb", round(c["mib"] / 1024 / 2))   # lend half the shared RAM
    res = new.setdefault("residents", {})
    known = {r["match"] for r in fleet.residents(new)}
    for n, u in use.items():
        for label in u.get("jobs") or []:
            proc = label.rpartition(" ")[0]
            for t, pre in fleet.PRESETS.items():
                if proc == pre["match"] and proc not in known:
                    res.setdefault(n, []).append({"type": t})
                    known.add(proc)
                    break
    text = json.dumps(new, indent=2) + "\n"
    if any(isinstance(r, dict) and "vram_gb" not in r and "vram_frac" not in fleet.PRESETS.get(r.get("type"), {})
           for rs in res.values() for r in rs):
        print("gpuq: add \"vram_gb\" to each resident so `where` and --vram count what it holds",
              file=sys.stderr)
    if args.print:
        print(text, end="")
        return 0
    if CONFIG.exists() and not args.force:
        print(text, end="")
        print("gpuq: %s exists — not changed (the merged config is above; --force writes it, keeping a .bak)"
              % CONFIG, file=sys.stderr)
        return 1
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG.exists():
        shutil.copy2(CONFIG, CONFIG.with_suffix(".json.bak"))
    CONFIG.write_text(text)
    print("wrote %s — %d card(s): %s. Edit reserve_gb / vram_gb / tags / residents to taste; "
          "`gpuq inventory` shows the result." % (CONFIG, len(cards), ", ".join(cards) or "none found"))
    return 0


def want_priority(args, q, e=None):
    """The priority the flags ask for, or None when none was given. --preempt = one above the
    running job of line Q (without a line, e.g. `where`: above anything)."""
    lvl = getattr(args, "level", None)
    val = getattr(args, "priority", None)
    if val is not None:
        val = str(val).strip().lower()
        if val.lstrip("-").isdigit():
            return int(val)
        if val not in PRIORITY and val != "preempt":
            raise QueueError(2, "--priority: %r is not a number or one of %s, preempt"
                             % (val, ", ".join(PRIORITY)))
        lvl = val
    if lvl == "preempt":
        if q is None:
            return 10 ** 6
        r = next((x for x in q if x["state"] in ("running", "yielding") and x is not e), None)
        return max(prio(r) + 1, PRIORITY["normal"]) if r else PRIORITY["normal"]
    return PRIORITY.get(lvl) if lvl else None


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
        return note
    return None


# ---- evacuate / restore: move a card's resident services off it, to free the whole card ----

def _unit_for(proc, cfg):
    """The systemd user unit for a resident process name (config `units`, else <proc>.service)."""
    for r in fleet.residents(cfg):
        if proc in (r["name"], r["match"]):
            return r["unit"]
    return (cfg.get("units") or {}).get(proc) or (proc + ".service")


def _units_on(card, cfg):
    return [r["unit"] for r in fleet.residents(cfg) if r["card"] == card]


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
    units = _units_on(card, cfg)
    if not units:
        raise QueueError(2, "no resident services configured on the %s (see the config file, 'residents')"
                         % card)
    return card, to, cards[to].get("env_value") or cards[to]["uuid"], units


def _systemctl(*a):
    return subprocess.run(["systemctl", "--user", *a], capture_output=True, text=True)


def _restart_all(units, verb="restart"):
    """daemon-reload, then restart each unit; returns 0, or 1 if any restart failed (printed)."""
    _systemctl("daemon-reload")
    rc = 0
    for u in units:
        r = _systemctl(verb, u)
        if r.returncode != 0:
            rc = 1
            print("  ! %s %s failed: %s" % (verb, u, (r.stderr or "").strip()), file=sys.stderr)
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
    freed = read_freed()
    stopped = [u for u, c in freed.items() if not args.card or c == args.card]
    card = args.card or (st or {}).get("card") or cfg.get("default")
    units = []
    if st or not stopped:      # an evac to undo (or, with nothing freed, the old default)
        units = (st or {}).get("units") or _units_on(card, cfg)
    if not units and not stopped:
        raise QueueError(2, "nothing to restore")
    if units:
        print("restore: put residents back on the %s" % card)
        for u in units:
            print("  %s → drop the override, then restart" % u)
    for u in stopped:
        print("  %s → start again (stopped by `gpuq free`)" % u)
    if args.dry_run:
        print("(dry run — nothing changed)")
        return 0
    rc = 0
    if units:
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
    if stopped:
        rc = _restart_all(stopped, "start") or rc
        write_freed({u: c for u, c in freed.items() if u not in stopped})
    return rc


# ---- free: release a card's resident memory without moving anything ----

def free_card(card, cfg, dry_run=False, say=print):
    """Release what the residents of CARD hold: unload models (ollama, llama-swap), stop units
    (`free: "stop"`, vLLM, llama-server) or run the resident's own `free` shell command.
    Returns (rc, units stopped). Residents without a way to free them are reported, not touched."""
    rc, stopped = 0, []
    here = [r for r in fleet.residents(cfg) if r["card"] == card]
    if not here:
        say("free: no residents configured on the %s" % card)
        return 0, []
    for r in here:
        how = r.get("free")
        if not how:
            say("  %s: no way to free it declared (set \"type\" or \"free\") — left alone" % r["name"])
            continue
        if how in ("ollama", "llama-swap"):
            say("  %s: unload models over HTTP (%s)" % (r["name"], r.get("url")))
            if dry_run:
                continue
            try:
                got = (fleet.ollama_unload if how == "ollama" else fleet.llamaswap_unload)(r["url"])
                say("    unloaded: %s" % (", ".join(got) or "nothing was loaded"))
            except (OSError, ValueError) as err:
                rc = 1
                say("    ! %s: %s" % (r["name"], err))
        elif how == "stop":
            say("  %s: stop %s (restore starts it)" % (r["name"], r["unit"]))
            if dry_run:
                continue
            res = _systemctl("stop", r["unit"])
            if res.returncode:
                rc = 1
                say("    ! stop failed: %s" % (res.stderr or "").strip())
            else:
                stopped.append(r["unit"])
        else:
            say("  %s: run %s" % (r["name"], how))
            if not dry_run and subprocess.run(how, shell=True).returncode:
                rc = 1
    if stopped:
        f = read_freed()
        f.update({u: card for u in stopped})
        write_freed(f)
    return rc, stopped


def _free(args, now):
    cfg = config()
    card = args.card or default_card(gpu_cards())
    print("free: release the residents' memory on the %s%s" % (card, " (dry run)" if args.dry_run else ""))
    rc, _ = free_card(card, cfg, args.dry_run)
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