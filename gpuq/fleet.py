"""gpuq fleet: what a host offers, and where a job fits.

The inventory is declarative. Cards are detected; the config can cap or reserve memory on a card,
tag it, and declare the always-on services (residents) that live on it and how much they hold.
gpuq never inspects those services' own configs: what you declare is what it plans with.

Pure functions here (no clock, no state files), plus the few HTTP calls that unload models from
a resident inference server (`free_resident`).
"""
import json
import statistics
import urllib.error
import urllib.request

from .core import join, jump, remaining

# Resident presets: `{"type": "ollama"}` in the config fills these in. Everything is overridable.
#   match     process name as gpuq sees it (nvidia-smi / fdinfo), used to recognise the resident
#   unit      systemd user unit (evac restarts it pinned to another card; free may stop it)
#   free      how `gpuq free` releases its memory: "ollama" / "llama-swap" unload over HTTP,
#             "stop" stops the unit (restore starts it again), or a shell command
#   vram_frac share of the card it grabs when no vram_gb is given (vLLM pre-allocates)
PRESETS = {
    "ollama": {"match": "ollama", "unit": "ollama.service", "free": "ollama",
               "url": "http://127.0.0.1:11434"},
    "llama-server": {"match": "llama-server", "unit": "llama-server.service", "free": "stop"},
    # llama-swap runs llama-server children: same process name, so `init` guesses llama-server
    "llama-swap": {"match": "llama-server", "unit": "llama-swap.service", "free": "llama-swap",
                   "url": "http://127.0.0.1:8080"},
    "vllm": {"match": "vllm", "unit": "vllm.service", "free": "stop", "vram_frac": 0.9},
}

VENDOR_TAG = {"amd": "rocm"}


def residents(cfg):
    """Every declared resident as a dict: name, card, match, unit, free, url, vram_gb, vram_frac.
    Config `residents` is {card: [entry]}, an entry being a process name (the old form) or a dict
    with `name` and/or `type`, plus any field above."""
    out = []
    units = cfg.get("units") or {}
    for card, items in (cfg.get("residents") or {}).items():
        for it in items or []:
            r = {"name": it} if isinstance(it, str) else dict(it)
            preset = PRESETS.get(r.get("type") or "", {})
            for k, v in preset.items():
                r.setdefault(k, v)
            r.setdefault("name", r.get("type") or r.get("match") or "?")
            r.setdefault("match", r["name"])
            r.setdefault("unit", units.get(r["name"]) or units.get(r["match"]) or r["name"] + ".service")
            r["card"] = card
            out.append(r)
    return out


def resident_names(cfg):
    """{card: [process names]}: the shape card_usage() matches running processes against."""
    out = {}
    for r in residents(cfg):
        out.setdefault(r["card"], []).append(r["match"])
    return out


def resident_mib(r, card_mib):
    if r.get("vram_gb") is not None:
        return int(float(r["vram_gb"]) * 1024)
    if r.get("vram_frac") is not None and card_mib:
        return int(card_mib * float(r["vram_frac"]))
    return 0


def card_tags(name, c, cfg):
    tags = set((((cfg.get("cards") or {}).get(name) or {}).get("tags")) or [])
    tags.add(VENDOR_TAG.get(c.get("vendor"), "cuda"))
    if c.get("integrated"):
        tags.add("igpu")
    return sorted(tags)


def line_wait(q, now, est_min=None, important=False, priority=None):
    """Seconds until a job joining line Q now could start, or None when an estimate in the way is
    missing. With est_min, the short-job-first jump is applied, as `join` would."""
    q = [dict(e) for e in q]
    probe = join(q, now, "?", "?", est_min=est_min, important=important,
                 taken={e["id"] for e in q}, priority=priority)
    if est_min:
        jump(q, now, probe["id"])
    wait = 0
    for e in q:
        if e is probe or e["id"] == probe["id"]:
            return wait
        left = remaining(e, now)
        if left is None:
            return None
        wait += left
    return wait


def inventory(cards, cfg, lines, now, use=None, evac=None, freed=()):
    """{card: {...}} — what each card offers. usable_mib = memory (capped by `vram_gb`) minus
    `reserve_gb` minus the residents living there now (an evac moves them, a free drops the ones
    it stopped). Cards that only exist as a queue line (no detection) have mib None."""
    ccfg = cfg.get("cards") or {}
    res = residents(cfg)
    out = {}
    names = list(cards) + [n for n in lines if n not in cards]
    for n in names:
        c = cards.get(n) or {}
        s = ccfg.get(n) or {}
        total = c.get("mib")
        cap = total
        if total is not None and s.get("vram_gb") is not None:
            cap = min(total, int(float(s["vram_gb"]) * 1024))
        reserve = int(float(s.get("reserve_gb") or 0) * 1024)
        here = []
        for r in res:
            home = r["card"]
            if evac and evac.get("card") == home and r["unit"] in (evac.get("units") or []):
                home = evac.get("to")
            if home == n and r["unit"] not in freed:
                here.append(r)
        rmib = sum(resident_mib(r, total) for r in here)
        freeable = sum(resident_mib(r, total) for r in here if r.get("free"))
        q = lines.get(n) or []
        run = next((e for e in q if e["state"] in ("running", "yielding")), None)
        u = (use or {}).get(n) or {}
        out[n] = {
            "vendor": c.get("vendor") or ("nvidia" if c else None),
            "integrated": bool(c.get("integrated")),
            "mib": total, "cap_mib": cap, "reserve_mib": reserve, "resident_mib": rmib, "freeable_mib": freeable,
            "usable_mib": None if cap is None else max(cap - reserve - rmib, 0),
            "tags": card_tags(n, c, cfg) if c else sorted(s.get("tags") or []),
            "residents": [r["name"] for r in here],
            "in_line": len(q),
            "running": {"id": run["id"], "owner": run["owner"], "job": run["job"]} if run else None,
            "wait_sec": line_wait(q, now),
            "jobs": u.get("jobs"), "busy": u.get("busy"),
        }
    return out


def gb(mib):
    """1536 -> "1.5 GB", 24576 -> "24 GB"."""
    return "?" if mib is None else "%g GB" % round(mib / 1024, 1)


def fits(card, vram_gb=None, tags=()):
    """(ok, reason) for a job needing VRAM_GB and TAGS on an inventory card."""
    missing = [t for t in tags or () if t not in card["tags"]]
    if missing:
        return False, "lacks tag %s" % ",".join(missing)
    if vram_gb:
        if card["usable_mib"] is None:
            return False, "memory unknown"
        if vram_gb * 1024 > card["usable_mib"]:
            why = "%s usable" % gb(card["usable_mib"])
            more = card["usable_mib"] + card.get("freeable_mib", 0)
            if card.get("freeable_mib") and vram_gb * 1024 <= more:
                why += "; %s after `gpuq free`" % gb(more)
            return False, why
    return True, ""


def rank(inv, lines, now, vram_gb=None, tags=(), est_min=None, important=False, prefer=None,
         priority=None):
    """[(name, ok, reason, wait_sec)] best first: fitting cards, then the shortest wait (with the
    short-job jump applied), then the preferred card, then the smallest card that fits (keep the
    big one free)."""
    rows = []
    for n, c in inv.items():
        ok, why = fits(c, vram_gb, tags)
        wait = line_wait(lines.get(n) or [], now, est_min, important, priority)
        rows.append((n, ok, why, wait))
    big = float("inf")
    rows.sort(key=lambda r: (not r[1], big if r[3] is None else r[3], r[0] != prefer,
                             inv[r[0]]["usable_mib"] or 0))
    return rows


def run_seconds(records):
    """{id: (job, owner, seconds)} for jobs that ran to `done`, summing their running stretches
    (a yield ends a stretch, the next start opens one). From history.jsonl records."""
    open_at, total, out = {}, {}, {}
    for h in records:
        eid, ev = h.get("id"), h.get("event")
        if ev == "started":
            open_at[eid] = h["t"]
        elif ev in ("yielded", "done", "purged", "removed") and eid in open_at:
            total[eid] = total.get(eid, 0) + h["t"] - open_at.pop(eid)
        if ev == "done" and eid in total:
            out[eid] = (h.get("job") or "", h.get("owner") or "", total.pop(eid))
        elif ev in ("purged", "removed"):
            total.pop(eid, None)
    return out


def estimate(records, job, owner=None, last=5):
    """(minutes, n) — the median of the last LAST completed runs of the same job (same text,
    case-insensitive; same owner preferred), or None if it never finished before."""
    key = job.strip().lower()
    runs = [(o, s) for j, o, s in run_seconds(records).values() if j.strip().lower() == key]
    if owner and any(o == owner for o, _ in runs):
        runs = [(o, s) for o, s in runs if o == owner]
    secs = [s for _, s in runs][-last:]
    if not secs:
        return None
    minutes = max(1, int(-(-statistics.median(secs) // 60)))
    return min(minutes, 10 ** 6), len(secs)


# ---- releasing a resident's memory over HTTP ----

def _http(method, url, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw.strip().startswith((b"{", b"[")) else raw.decode(errors="replace")


def ollama_unload(url):
    """Unload every model an Ollama server has loaded: keep_alive 0 per model in /api/ps.
    Returns the model names. Ollama loads them again on the next request."""
    url = url.rstrip("/")
    loaded = [m.get("name") or m.get("model") for m in (_http("GET", url + "/api/ps").get("models") or [])]
    for m in loaded:
        _http("POST", url + "/api/generate", {"model": m, "keep_alive": 0}, timeout=60)
    return loaded


def llamaswap_unload(url):
    """Ask llama-swap to unload all running models (POST /api/models/unload; older: GET /unload)."""
    url = url.rstrip("/")
    try:
        _http("POST", url + "/api/models/unload")
    except urllib.error.HTTPError as err:
        if err.code not in (404, 405):
            raise
        _http("GET", url + "/unload")
    return ["all"]
