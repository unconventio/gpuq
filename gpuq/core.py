"""gpuq core: the pure queue operations. No I/O, no clock: callers pass `now`.

A queue is a list of entry dicts; position = index + 1. At most one entry is
"running" (always #1); "yielding" means it was asked to step aside and still holds the card.
"""
import secrets

STALE_SEC = 15 * 60        # a running holder silent this long is purged
HEAD_GRACE_SEC = 15 * 60   # a waiting #1 must start within this long, or it is skipped
# Short jobs go first: a job of at most JUMP_MAX_SEC goes ahead of any job with JUMP_RATIO
# times as much left, unless that job is important (priority SHIELD or more) or outranks it.
JUMP_RATIO = 10
JUMP_MAX_SEC = 6 * 3600

# Priority: any integer, higher goes first. The named levels are the flags' defaults.
PRIORITY = {"chill": 5, "normal": 10, "important": 20, "ablaze": 30}
SHIELD = PRIORITY["important"]


def prio(e):
    """An entry's priority. Entries from before priorities: important = 20, else 10."""
    p = e.get("priority")
    if p is None:
        return PRIORITY["important"] if e.get("important") else PRIORITY["normal"]
    return p


def prio_mark(e):
    """One cell for the views: ⚑ ablaze, ★ important, ☾ chill, blank for normal."""
    p = prio(e)
    return "⚑" if p >= PRIORITY["ablaze"] else "★" if p >= SHIELD else "☾" if p < PRIORITY["normal"] else " "


class QueueError(Exception):
    def __init__(self, code, msg, save=False):
        super().__init__(msg)
        self.code = code
        self.save = save    # the queue changed before the refusal: write it anyway


# ---- pure queue operations: q is a list of entries, position = index + 1 ----

def find(q, ref, by_id=False):
    """Index of an entry by id, or (unless by_id) by 1-based position."""
    for i, e in enumerate(q):
        if e["id"] == ref:
            return i
    if not by_id and str(ref).isdigit() and 1 <= int(ref) <= len(q):
        return int(ref) - 1
    raise QueueError(4, "no entry %r in the queue" % ref)


def running(q):
    return q[0] if q and q[0]["state"] == "running" else None


def yielding(q):
    """Entries asked to give up the GPU that haven't stopped yet — they still hold the card."""
    return [e for e in q if e["state"] == "yielding"]


def normalize(q, now):
    """Only a waiting #1 carries head_since — the clock for its 15 minutes to start.
    A yielder that is #1 again (moved back, or everyone above it left) just keeps running."""
    if q and q[0]["state"] == "yielding":
        q[0].update(state="running", yield_since=None)
    for i, e in enumerate(q):
        if i == 0 and e["state"] == "waiting":
            e["head_since"] = e.get("head_since") or now
        else:
            e["head_since"] = None


def join(q, now, owner, job, why="", models=(), est_min=None, max_min=None, important=False,
         taken=(), priority=None):
    ids = {e["id"] for e in q} | set(taken)     # taken: ids in the other cards' lines
    eid = ""
    # Ids always carry a letter so they can never be mistaken for a position.
    while not eid or eid in ids or eid.isdigit():
        eid = secrets.token_hex(2)
    e = {"id": eid, "owner": owner, "job": job, "why": why, "models": list(models),
         "est_min": est_min, "max_min": max_min, "progress": "", "state": "waiting",
         "joined": now, "started": None, "heartbeat": None, "head_since": None, "skips": 0,
         "important": important}
    if priority is not None:
        e["priority"] = int(priority)
        e["important"] = e["priority"] >= SHIELD
    q.append(e)
    normalize(q, now)
    place(q, now, eid)
    return e


def place(q, now, ref):
    """Put an entry just behind the last other entry of at least its priority (first if none):
    a higher priority goes ahead, equals keep their order. Passing the running job asks it to
    yield. Hand-made order among the rest is left alone. Returns the new index."""
    i = find(q, ref, by_id=True)
    e, p = q[i], prio(q[i])
    others = [x for x in q if x is not e]
    j = max((k + 1 for k, x in enumerate(others) if prio(x) >= p), default=0)
    if j != i:
        move(q, now, e["id"], str(j + 1))
    return j


def remaining(e, now):
    """Estimated seconds of GPU work left, or None without an estimate."""
    if not e.get("est_min"):
        return None
    if e["state"] in ("running", "yielding") and e.get("started"):
        return max(e["est_min"] * 60 - (now - e["started"]), 0)
    return e["est_min"] * 60


def jump(q, now, ref):
    """Move a short job ahead of the long jobs directly in front of it; it stops at the
    first job that is important, outranks it, has no estimate, or isn't JUMP_RATIO times longer.
    Passing the running job asks it to yield. Returns the entries passed."""
    i = find(q, ref, by_id=True)
    mine, mp = remaining(q[i], now), prio(q[i])
    if mine is None or mine > JUMP_MAX_SEC:
        return []
    j = i
    while j > 0:
        x = q[j - 1]
        left = remaining(x, now)
        if prio(x) >= SHIELD or prio(x) > mp or left is None or left < JUMP_RATIO * max(mine, 60):
            break
        j -= 1
    passed = q[j:i]
    if passed:
        move(q, now, ref, str(j + 1))
    return passed


def start(q, now, eid):
    i = find(q, eid, by_id=True)
    e = q[i]
    if e["state"] == "running":
        return e
    if e["state"] == "yielding":
        # Back at the start loop means its GPU work has stopped: the yield is done.
        e.update(state="waiting", started=None, heartbeat=None, yield_since=None)
        normalize(q, now)
        raise QueueError(3, "%s yielded — now #%d, keep polling start" % (eid, i + 1), save=True)
    if i == 0 and yielding(q):
        y = yielding(q)[0]
        raise QueueError(3, "%s (%s) is still stopping its job to yield — retry in a minute"
                         % (y["owner"], y["id"]))
    if i != 0:
        raise QueueError(3, "%s is #%d, not #1 — wait your turn" % (eid, i + 1))
    e.update(state="running", started=now, heartbeat=now, skips=0)
    normalize(q, now)
    return e


def beat(q, now, eid, progress=None):
    e = q[find(q, eid, by_id=True)]
    e["heartbeat"] = now
    if progress is not None:
        e["progress"] = progress
    if e["state"] == "yielding":
        raise QueueError(5, "yield: stop your GPU work, then poll `gpuq start %s` until it is"
                         " your turn again" % eid, save=True)
    return e


def done(q, now, eid):
    e = q.pop(find(q, eid, by_id=True))
    normalize(q, now)
    return e


def move(q, now, src, dst):
    i = find(q, src)
    if not str(dst).isdigit() or not 1 <= int(dst) <= len(q):
        raise QueueError(2, "position %s is out of range 1..%d" % (dst, len(q)))
    j = int(dst) - 1
    # Moving the running job down, or anything above it, asks it to yield: it keeps the
    # card (state "yielding") until it stops. Moving a yielder back to #1 calls the yield off.
    e = q.pop(i)
    if e["state"] == "running" and j > 0:
        e.update(state="yielding", yield_since=now)
    if j == 0 and q and q[0]["state"] == "running":
        q[0].update(state="yielding", yield_since=now)
    q.insert(j, e)
    normalize(q, now)
    return q[j]


def remove(q, now, ref):
    e = q.pop(find(q, ref))
    normalize(q, now)
    return e


def sweep(q, now, paused=False):
    """Garbage collection. Returns [(event, entry, note)]. A pause stops the no-show clock
    (nobody may start, so nobody can be late) but a silent runner is still purged."""
    events = []
    r = running(q)
    if r:
        silent = now - (r["heartbeat"] or r["started"])
        if silent > STALE_SEC:
            q.pop(0)
            events.append(("purged", r, "no heartbeat for %s" % fmt_dur(silent)))
        elif r.get("max_min") and now - r["started"] > r["max_min"] * 60:
            q.pop(0)
            events.append(("purged", r, "ran past its max of %s" % fmt_dur(r["max_min"] * 60)))
    held = bool(yielding(q))    # a yield (even one purged just now) held the line until now
    for y in yielding(q):
        silent = now - (y["heartbeat"] or y["started"])
        if silent > STALE_SEC:
            q.remove(y)
            events.append(("purged", y, "yielding, no heartbeat for %s" % fmt_dur(silent)))
    normalize(q, now)
    h = q[0] if q else None
    if paused or held:          # nobody may start, so nobody can be late
        if h and h["state"] == "waiting":
            h["head_since"] = now
        return events
    if h and h["state"] == "waiting" and now - h["head_since"] > HEAD_GRACE_SEC:
        q.pop(0)
        if h.get("skips", 0) >= 1:
            events.append(("removed", h, "no-show at #1 twice"))
        else:
            h["skips"] = h.get("skips", 0) + 1
            h["head_since"] = None
            q.append(h)
            events.append(("skipped", h, "didn't start within 15 min at #1"))
        normalize(q, now)
    return events


def fmt_dur(sec):
    sec = int(sec)
    if sec < 60:
        return "<1m"
    h, m = divmod(sec // 60, 60)
    return "%dh%02d" % (h, m) if h else "%dm" % m
