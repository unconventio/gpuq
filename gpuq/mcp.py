"""gpuq over MCP: the queue as tools for AI agents. Stdio, newline-delimited JSON-RPC 2.0, stdlib only.

Run `gpuq mcp` (or `gpuq-mcp`) as an MCP server command. Every tool is a thin wrapper over the CLI
with --json, so the agent sees the same states a script does: start -> go / wait / gone,
beat -> ok / yield / purged.

Agents forget to heartbeat. So any tool call from this session also beats every job the session
has started (proof of life), and a pending yield for one of them comes back on *any* result as a
`notices` entry. A session that makes no calls at all still goes stale and is purged after 15 min.
For long work, `gpuq run ID -- CMD` in a shell beats by itself; MCP is for asking and holding.
"""
import contextlib
import io
import json
import shlex
import subprocess
import sys

from . import __version__

PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

INSTRUCTIONS = (
    "gpuq is a numbered queue per GPU card, per machine. To use a GPU: gpuq_where to find a card "
    "(optional), gpuq_join to get in line (keep the id), poll gpuq_start every ~60 s until state is "
    "'go', then run your work with the returned env (CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES) "
    "and call gpuq_beat at least every 10 minutes. If a beat or any result says 'yield', stop your "
    "GPU work and go back to polling gpuq_start: you resume when you are #1 again. Call gpuq_done "
    "when finished, with exit_code. If a job of yours dies, is purged or stalls, a notice comes back "
    "on your next tool call (and via gpuq_inbox): read it, check the log, tell your human. "
    "Never use a GPU without a 'go'.")

_HOST = {"host": {"type": "string", "description": "machine to act on (ssh); default: this one"}}
_ID = {"id": {"type": "string", "description": "the id gpuq_join returned"}}
_NEED = {
    "vram_gb": {"type": "number", "description": "GPU memory the job needs, GB"},
    "tags": {"type": "array", "items": {"type": "string"},
             "description": "card tags required: cuda, rocm, igpu, or ones the owner defined"},
    "est_min": {"type": "integer", "description": "expected minutes of GPU time"},
}

TOOLS = [
    {"name": "gpuq_inventory",
     "description": "What every card on every configured machine offers: usable memory (after reserve and "
                    "resident servers), tags, residents, queue length, current job, wait.",
     "inputSchema": {"type": "object", "properties": dict(
         hosts={"type": "string", "description": "comma-separated machines; default: all configured"})}},
    {"name": "gpuq_where",
     "description": "Which card, on which machine, can take a job needing this memory/tags, best first, with "
                    "the estimated wait (short jobs may jump long ones). Join on the first row that fits.",
     "inputSchema": {"type": "object", "properties": dict(_NEED, hosts={"type": "string"})}},
    {"name": "gpuq_show",
     "description": "The queue on one machine: every card's line, in order, with states.",
     "inputSchema": {"type": "object", "properties": dict(_HOST)}},
    {"name": "gpuq_join",
     "description": "Get in line for a GPU. Without gpu, a card is picked from vram_gb/tags (shortest wait), "
                    "else the owner's or host's default. Without est_min, past runs of the same job are used. "
                    "Returns the id, card and position.",
     "inputSchema": {"type": "object", "required": ["owner", "job"], "properties": dict(
         _NEED, **_HOST,
         owner={"type": "string", "description": "who you are (session or project name)"},
         job={"type": "string", "description": "what the job is; reuse the same text for repeat jobs"},
         why={"type": "string"},
         models={"type": "array", "items": {"type": "string"}},
         max_min={"type": "integer", "description": "minutes after which you are purged"},
         priority={"type": "integer", "description": "higher goes first; normal 10, chill 5, important 20,"
                   " ablaze 30. Outranking the running job makes it yield"},
         level={"type": "string", "enum": ["chill", "normal", "important", "ablaze", "preempt"],
                "description": "a named priority; preempt = one above the running job's"},
         important={"type": "boolean", "description": "same as level important"},
         contact={"type": "string", "description": "who is told if the job dies, is purged or stalls "
                  "(comma-separated; default: owner). Notices come back on your tool calls and via gpuq_inbox"},
         desktop={"type": "boolean", "description": "also pop a desktop notice for the human. Only when "
                  "they asked for it"},
         log={"type": "string", "description": "path of the job's log file, quoted in notices"},
         gpu={"type": "string", "description": "a specific card by name (see gpuq_inventory)"})}},
    {"name": "gpuq_start",
     "description": "Try to take the GPU. state 'go': run now with env. 'wait': not your turn (or a yield was "
                    "acknowledged), call again in ~60 s. 'gone': the id is no longer in line.",
     "inputSchema": {"type": "object", "required": ["id"], "properties": dict(_ID, **_HOST)}},
    {"name": "gpuq_beat",
     "description": "Heartbeat while you hold the GPU (at least every 10 min). state 'yield': stop your GPU "
                    "work now and poll gpuq_start. 'purged': you lost your place; stop.",
     "inputSchema": {"type": "object", "required": ["id"], "properties": dict(
         _ID, **_HOST, progress={"type": "string", "description": "e.g. 41/80"})}},
    {"name": "gpuq_done",
     "description": "Release the GPU and leave the line. Pass exit_code: non-zero (or ending within 10 "
                    "minutes without one) tells the job's contact.",
     "inputSchema": {"type": "object", "required": ["id"], "properties": dict(
         _ID, **_HOST, exit_code={"type": "integer", "description": "the job's exit code; 0 = success"})}},
    {"name": "gpuq_inbox",
     "description": "Notices for a contact: jobs that died (exit code, early death), were purged (no "
                    "heartbeat) or stalled, with the log path. Unread ones are marked read.",
     "inputSchema": {"type": "object", "required": ["contact"], "properties": dict(
         _HOST, contact={"type": "string", "description": "the contact (usually your owner name)"},
         all={"type": "boolean", "description": "every notice kept, read or not"})}},
]


def run_cli(args, host=None):
    """(exit code, stdout, stderr) of `gpuq ARGS`, here in-process or on HOST over ssh."""
    from . import cli
    if host and host != cli.HOST:
        try:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                                cli.remote_cmd() + " " + shlex.join(args)],
                               capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return 1, "", "%s: no answer within 60 s" % host
        return r.returncode, r.stdout, r.stderr
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main(list(args))
        except SystemExit as e:        # argparse
            rc = e.code if isinstance(e.code, int) else 2
    return rc, out.getvalue(), err.getvalue()


def _json_of(rc, out, err):
    try:
        return json.loads(out)
    except ValueError:
        return {"ok": False, "code": rc, "state": "error",
                "message": (err.strip() or out.strip() or "no output").splitlines()[-1]}


class Server:
    def __init__(self, cli_runner=run_cli):
        self.run = cli_runner
        self.held = {}      # id -> host: jobs this session started, beaten on every call
        self.watch = set()  # (contact, host): inboxes checked on every call, from this session's joins

    # ---- tools ----
    def call(self, name, a):
        host = a.get("host")
        if name == "gpuq_inventory":
            return self._plain(["inventory", "--json"] + (["--hosts", a["hosts"]] if a.get("hosts") else []))
        if name == "gpuq_where":
            return self._plain(["where", "--json"] + self._need(a) +
                               (["--hosts", a["hosts"]] if a.get("hosts") else []))
        if name == "gpuq_show":
            rc, out, err = self.run(["show", "--json"], host)
            snap = _json_of(rc, out, err)
            if "lines" in snap:
                snap = {"host": snap.get("host"), "default": snap.get("default"),
                        "paused": bool(snap.get("paused")),
                        "lines": {n: [{"position": i + 1, "id": e["id"], "owner": e["owner"], "job": e["job"],
                                       "state": e["state"], "est_min": e.get("est_min"),
                                       "vram_gb": e.get("vram_gb"), "progress": e.get("progress")}
                                      for i, e in enumerate(q)] for n, q in snap["lines"].items()}}
            return snap
        if name == "gpuq_join":
            spec = {k: a[k] for k in ("owner", "job", "why", "models", "max_min", "important", "gpu", "priority", "level",
                                      "vram_gb", "tags", "est_min") if a.get(k) not in (None, "", [])}
            args = ["join", "--json", "--owner", spec.pop("owner"), "--job", spec.pop("job")]
            for k, flag in (("why", "--why"), ("gpu", "--gpu"), ("max_min", "--max"), ("est_min", "--est"),
                            ("vram_gb", "--vram")):
                if k in spec:
                    args += [flag, str(spec[k])]
            if spec.get("models"):
                args += ["--models", ",".join(spec["models"])]
            for t in spec.get("tags") or []:
                args += ["--tag", t]
            if "priority" in spec:
                args += ["--priority", str(spec["priority"])]
            elif spec.get("level"):
                args.append("--" + spec["level"])
            elif spec.get("important"):
                args.append("--important")
            if a.get("contact"):
                args += ["--contact", a["contact"]]
            if a.get("desktop"):
                args.append("--desktop")
            if a.get("log"):
                args += ["--log", a["log"]]
            res = _json_of(*self.run(args, host))
            if res.get("ok"):
                for c in (a.get("contact") or a["owner"]).split(","):
                    if c.strip():
                        self.watch.add((c.strip(), host))
            return res
        if name == "gpuq_inbox":
            if not a.get("contact"):
                return {"ok": False, "state": "error", "message": "contact is required"}
            args = ["inbox", "--json", "--contact", a["contact"]] + (["--all"] if a.get("all") else ["--ack"])
            if host:
                args += ["--hosts", host]
            host = None     # inbox reaches other machines itself
            rc, out, err = self.run(args, host)
            try:
                return {"ok": True, "notices": json.loads(out)}
            except ValueError:
                return {"ok": False, "state": "error", "message": (err.strip() or "no output").splitlines()[-1]}
        if name in ("gpuq_start", "gpuq_beat", "gpuq_done"):
            if not a.get("id"):
                return {"ok": False, "state": "error", "message": "id is required"}
            verb = name[5:]
            args = [verb, a["id"], "--json"] + (["--progress", a["progress"]]
                                                if verb == "beat" and a.get("progress") else [])
            if verb == "done" and a.get("exit_code") is not None:
                args.append("--exit=%d" % int(a["exit_code"]))
            res = _json_of(*self.run(args, host))
            st = res.get("state")
            if verb == "start" and st == "go":
                self.held[a["id"]] = host
                res["next"] = "run your GPU work now with env; call gpuq_beat at least every 10 minutes"
            elif verb == "start" and st == "wait":
                res["next"] = "call gpuq_start again in about 60 seconds"
            elif st == "yield":
                res["next"] = "stop your GPU work now, then poll gpuq_start; you resume when #1 again"
            if verb == "done" or st in ("gone", "purged"):
                self.held.pop(a["id"], None)
            return res
        raise KeyError(name)

    def _need(self, a):
        out = []
        if a.get("vram_gb") is not None:
            out += ["--vram", str(a["vram_gb"])]
        for t in a.get("tags") or []:
            out += ["--tag", t]
        if a.get("est_min") is not None:
            out += ["--est", str(a["est_min"])]
        return out

    def _plain(self, args):
        rc, out, err = self.run(args)
        try:
            return {"ok": True, "result": json.loads(out)}
        except ValueError:
            return {"ok": False, "state": "error", "message": (err.strip() or "no output").splitlines()[-1]}

    def proof_of_life(self, skip=None):
        """Beat every held job except SKIP; return notices for yields and purges."""
        notices = []
        for eid, host in list(self.held.items()):
            if eid == skip:
                continue
            res = _json_of(*self.run(["beat", eid, "--json"], host))
            if res.get("state") == "yield":
                notices.append({"id": eid, "state": "yield",
                                "message": "yield requested: stop the GPU work for %s and poll gpuq_start" % eid})
            elif res.get("state") in ("purged", "error") and res.get("code") == 4:
                self.held.pop(eid, None)
                notices.append({"id": eid, "state": "purged", "message": "%s is no longer in line: stop" % eid})
        for contact, host in sorted(self.watch, key=str):
            from . import cli
            rc, out, err = self.run(["inbox", "--json", "--ack", "--contact", contact,
                                     "--hosts", host or cli.HOST], host)
            try:
                got = json.loads(out)
            except ValueError:
                continue
            for n in got:
                notices.append({"id": n["id"], "state": n["kind"], "contact": contact,
                                "message": "%s (%s) %s: %s%s" % (n.get("job"), n["id"], n["kind"], n.get("text"),
                                                                "; log: %s" % n["log"] if n.get("log") else "")})
        return notices

    # ---- JSON-RPC ----
    def handle(self, msg):
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if mid is None:             # a notification: nothing to answer
            return None
        if method == "initialize":
            want = params.get("protocolVersion")
            return self._ok(mid, {"protocolVersion": want if want in PROTOCOLS else PROTOCOLS[0],
                                  "capabilities": {"tools": {"listChanged": False}},
                                  "serverInfo": {"name": "gpuq", "version": __version__},
                                  "instructions": INSTRUCTIONS})
        if method == "ping":
            return self._ok(mid, {})
        if method == "tools/list":
            return self._ok(mid, {"tools": TOOLS})
        if method == "tools/call":
            name, a = params.get("name"), params.get("arguments") or {}
            try:
                res = self.call(name, a)
            except KeyError:
                return self._err(mid, -32602, "unknown tool %r" % name)
            except Exception as err:   # a tool failure is a result, not a protocol error
                res = {"ok": False, "state": "error", "message": "%s: %s" % (type(err).__name__, err)}
            notices = self.proof_of_life(skip=a.get("id"))
            if notices:
                res["notices"] = notices
            failed = not res.get("ok", True) and res.get("state") == "error"
            return self._ok(mid, {"content": [{"type": "text", "text": json.dumps(res)}],
                                  "structuredContent": res, "isError": failed})
        return self._err(mid, -32601, "method not found: %s" % method)

    @staticmethod
    def _ok(mid, result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    @staticmethod
    def _err(mid, code, message):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def serve(stdin=None, stdout=None):
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    srv = Server()
    for line in stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            reply = srv.handle(msg) if isinstance(msg, dict) else None
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()
    return 0


def main():
    sys.exit(serve())
