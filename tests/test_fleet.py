"""Fleet features: inventory, placement (--vram / --tag / where), estimates from history, resident
presets and `free`, JSON in and out, and the MCP server. Every test uses its own temp state dir and
fake cards (GPUQ_CARDS); nothing here talks to a real GPU, queue, systemd or inference server."""
import http.server
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GPUQ_STATE", tempfile.mkdtemp(prefix="gpuq-test-"))
os.environ.setdefault("GPUQ_CONFIG", os.path.join(os.environ["GPUQ_STATE"], "no-config.json"))
os.environ.setdefault("GPUQ_COMPAT_LOCK", "")
from gpuq import cli as g, fleet, mcp  # noqa: E402

T0 = 1_760_000_000.0
CARDS = {"3090": {"uuid": "GPU-aaaa", "mib": 24576}, "3060": {"uuid": "GPU-bbbb", "mib": 12288}}
IGPU = {"880M": {"vendor": "amd", "uuid": "GPU-cc", "mib": 24576, "integrated": True,
                 "env_var": "ROCR_VISIBLE_DEVICES", "env_value": "0", "pci": "0000:c5:00.0"}}


def entry(eid, state="waiting", est=None, started=None, owner="o"):
    return {"id": eid, "owner": owner, "job": "j", "state": state, "est_min": est, "started": started,
            "heartbeat": started, "head_since": None, "important": False, "skips": 0}


class Inventory(unittest.TestCase):
    def test_reserve_cap_and_residents(self):
        cfg = {"cards": {"3090": {"reserve_gb": 2, "tags": ["fast"]}, "880M": {"vram_gb": 8}},
               "residents": {"3060": ["whisper", {"type": "ollama", "vram_gb": 6}]}}
        inv = fleet.inventory(dict(CARDS, **IGPU), cfg, {}, T0)
        self.assertEqual(inv["3090"]["usable_mib"], 22 * 1024)
        self.assertEqual(inv["3090"]["tags"], ["cuda", "fast"])
        self.assertEqual(inv["3060"]["usable_mib"], 6 * 1024)
        self.assertEqual(inv["3060"]["residents"], ["whisper", "ollama"])
        self.assertEqual(inv["880M"]["usable_mib"], 8 * 1024)
        self.assertEqual(inv["880M"]["tags"], ["igpu", "rocm"])

    def test_vllm_preset_takes_most_of_the_card(self):
        inv = fleet.inventory(CARDS, {"residents": {"3090": [{"type": "vllm"}]}}, {}, T0)
        self.assertEqual(inv["3090"]["usable_mib"], 24576 - int(24576 * 0.9))

    def test_evac_moves_resident_memory_and_free_drops_it(self):
        cfg = {"residents": {"3090": [{"type": "ollama", "vram_gb": 8}]}}
        evac = {"card": "3090", "to": "3060", "units": ["ollama.service"]}
        inv = fleet.inventory(CARDS, cfg, {}, T0, evac=evac)
        self.assertEqual((inv["3090"]["usable_mib"], inv["3060"]["usable_mib"]), (24576, 4096))
        inv = fleet.inventory(CARDS, cfg, {}, T0, freed={"ollama.service": "3090"})
        self.assertEqual(inv["3090"]["usable_mib"], 24576)

    def test_old_resident_form_still_matches_processes(self):
        cfg = {"residents": {"3060": ["whisper", {"type": "ollama"}]}, "units": {"whisper": "w.service"}}
        self.assertEqual(fleet.resident_names(cfg), {"3060": ["whisper", "ollama"]})
        self.assertEqual(g._unit_for("whisper", cfg), "w.service")
        self.assertEqual(g._unit_for("ollama", cfg), "ollama.service")

    def test_wait_applies_the_short_job_jump(self):
        q = [entry("a1", "running", est=600, started=T0), entry("b2", est=60)]
        self.assertEqual(fleet.line_wait(q, T0), 660 * 60)
        self.assertEqual(fleet.line_wait(q, T0, est_min=5), 0)         # 10x shorter than both: jumps both
        self.assertEqual(fleet.line_wait(q, T0, est_min=10), 660 * 60)  # not 10x shorter than b2: no jump
        self.assertIsNone(fleet.line_wait([entry("c3")], T0))          # no estimate in the way
        self.assertEqual(fleet.line_wait([], T0), 0)

    def test_rank_fits_then_wait_then_smallest(self):
        inv = fleet.inventory(CARDS, {}, {}, T0)
        rows = fleet.rank(inv, {}, T0, vram_gb=8)
        self.assertEqual([r[0] for r in rows], ["3060", "3090"])       # both free: keep the big one
        rows = fleet.rank(inv, {}, T0, vram_gb=16)
        self.assertEqual([(r[0], r[1]) for r in rows], [("3090", True), ("3060", False)])
        busy = {"3060": [entry("a1", "running", est=120, started=T0)]}
        self.assertEqual(fleet.rank(inv, busy, T0, vram_gb=8)[0][0], "3090")
        self.assertFalse(fleet.fits(inv["3090"], tags=["rocm"])[0])


class Estimates(unittest.TestCase):
    def rec(self, t, ev, eid, job="eval", owner="a"):
        return {"t": t, "event": ev, "id": eid, "job": job, "owner": owner}

    def test_median_of_completed_runs_with_yields(self):
        h = [self.rec(0, "started", "x1"), self.rec(600, "yielded", "x1"), self.rec(900, "started", "x1"),
             self.rec(1500, "done", "x1"),                                   # 20 min of GPU time
             self.rec(2000, "started", "x2"), self.rec(2000 + 40 * 60, "done", "x2"),
             self.rec(9000, "started", "x3"), self.rec(9100, "purged", "x3"),  # not a finish
             self.rec(9200, "started", "y1", job="other"), self.rec(9300, "done", "y1", job="other")]
        self.assertEqual(fleet.estimate(h, "EVAL "), (30, 2))
        self.assertEqual(fleet.estimate(h, "other"), (2, 1))
        self.assertIsNone(fleet.estimate(h, "never"))

    def test_same_owner_preferred(self):
        h = [self.rec(0, "started", "a1"), self.rec(600, "done", "a1"),
             self.rec(0, "started", "b1", owner="b"), self.rec(6000, "done", "b1", owner="b")]
        self.assertEqual(fleet.estimate(h, "eval", "b"), (100, 1))


class CLI(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gpuq-fleet-"))
        self.cfg = self.tmp / "config.json"

    def cli(self, *args, cards=CARDS, stdin=None, now=T0, cfg=None):
        if cfg is not None:
            self.cfg.write_text(json.dumps(cfg))
        env = dict(os.environ, GPUQ_STATE=str(self.tmp / "state"), GPUQ_CONFIG=str(self.cfg),
                   GPUQ_COMPAT_LOCK="", GPUQ_CARDS=json.dumps(cards), GPUQ_NOW=str(now),
                   PYTHONPATH=str(ROOT))
        return subprocess.run([sys.executable, "-m", "gpuq", *args], capture_output=True, text=True,
                              env=env, input=stdin)

    def js(self, *args, **kw):
        r = self.cli(*args, **kw)
        return json.loads(r.stdout), r

    def test_join_vram_picks_the_smallest_card_that_fits(self):
        out, _ = self.js("join", "--owner", "a", "--job", "j", "--vram", "8", "--json")
        self.assertEqual(out["gpu"], "3060")
        out, _ = self.js("join", "--owner", "a", "--job", "j", "--vram", "20", "--json")
        self.assertEqual(out["gpu"], "3090")

    def test_join_refuses_what_fits_nowhere(self):
        r = self.cli("join", "--owner", "a", "--job", "j", "--vram", "40")
        self.assertEqual(r.returncode, 2)
        self.assertIn("gpuq where --vram 40", r.stderr)
        r = self.cli("join", "--owner", "a", "--job", "j", "--vram", "16", "--gpu", "3060")
        self.assertEqual(r.returncode, 2)
        out, r = self.js("join", "--owner", "a", "--job", "j", "--tag", "rocm", "--json")
        self.assertEqual((out["ok"], out["state"], r.returncode), (False, "error", 2))

    def test_reserve_and_residents_change_the_choice(self):
        cfg = {"cards": {"3060": {"reserve_gb": 1}}, "residents": {"3060": [{"type": "ollama", "vram_gb": 4}]}}
        out, _ = self.js("join", "--owner", "a", "--job", "j", "--vram", "8", "--json", cfg=cfg)
        self.assertEqual(out["gpu"], "3090")      # 12 - 1 - 4 = 7 GB usable on the 3060

    def test_spec_from_stdin_and_flags_win(self):
        spec = {"owner": "agent", "job": "train", "vram_gb": 8, "tags": "cuda", "est": 30, "models": ["m1", "m2"]}
        out, r = self.js("join", "--spec", "-", "--json", "--est", "45", stdin=json.dumps(spec))
        self.assertEqual((out["gpu"], out["est_min"]), ("3060", 45), r.stderr)
        snap, _ = self.js("show", "--json")
        e = snap["lines"]["3060"][0]
        self.assertEqual((e["models"], e["vram_gb"], e["tags"]), (["m1", "m2"], 8, ["cuda"]))
        r = self.cli("join", "--spec", "-", stdin='{"owner": "a", "job": "j", "color": "red"}')
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown key", r.stderr)
        self.assertEqual(self.cli("join", "--owner", "a").returncode, 2)

    def test_json_states_for_start_and_beat(self):
        a, _ = self.js("join", "--owner", "a", "--job", "long", "--est", "600", "--gpu", "3090", "--json")
        st, _ = self.js("start", a["id"], "--json")
        self.assertEqual((st["state"], st["env"]), ("go", {"CUDA_VISIBLE_DEVICES": "GPU-aaaa"}))
        b, _ = self.js("join", "--owner", "b", "--job", "short", "--est", "10", "--gpu", "3090", "--json", now=T0 + 60)
        self.assertTrue(b["jumped"])
        st, r = self.js("start", b["id"], "--json", now=T0 + 61)
        self.assertEqual((st["state"], r.returncode), ("wait", 3))
        bt, r = self.js("beat", a["id"], "--json", now=T0 + 62)
        self.assertEqual((bt["state"], r.returncode), ("yield", 5))
        st, _ = self.js("start", "nope", "--json")
        self.assertEqual(st["state"], "gone")

    def test_estimate_from_history_on_join(self):
        j, _ = self.js("join", "--owner", "a", "--job", "nightly eval", "--json")
        self.cli("start", j["id"])
        self.cli("done", j["id"], now=T0 + 25 * 60)
        j, r = self.js("join", "--owner", "a", "--job", "nightly eval", "--json", now=T0 + 3600)
        self.assertEqual(j["est_min"], 25)
        self.assertIn("history", j["est_from"])
        self.assertIn("estimate from history", r.stderr)
        j, _ = self.js("join", "--owner", "a", "--job", "nightly eval", "--no-history", "--json")
        self.assertIsNone(j["est_min"])

    def test_where_json_and_text(self):
        self.cli("join", "--owner", "a", "--job", "j", "--est", "120", "--gpu", "3060")
        rows, r = self.js("where", "--vram", "10", "--local", "--json")
        self.assertEqual([(x["card"], x["fits"]) for x in rows], [("3090", True), ("3060", True)])
        self.assertEqual(rows[1]["wait_sec"], 120 * 60)
        r = self.cli("where", "--vram", "10", "--local")
        self.assertIn("→ gpuq join --gpu 3090 --vram 10", r.stdout)
        self.assertEqual(self.cli("where", "--vram", "99", "--local").returncode, 1)

    def test_inventory_text_and_json(self):
        cfg = {"cards": {"880M": {"vram_gb": 8}}}
        inv, _ = self.js("inventory", "--local", "--json", cards=IGPU, cfg=cfg)
        self.assertEqual(inv["880M"]["usable_mib"], 8192)
        out = self.cli("inventory", "--local", cards=IGPU).stdout
        self.assertIn("24 GB shared", out)
        self.assertIn("capped from 24 GB", out)
        self.assertIn("igpu", out)

    def test_init_prints_a_starter_config_and_does_not_overwrite(self):
        out, r = self.js("init", "--print", cards=dict(CARDS, **IGPU))
        self.assertEqual(out["default"], "3090")
        self.assertEqual(out["cards"]["880M"]["vram_gb"], 12)
        self.assertFalse(self.cfg.exists())
        self.assertEqual(self.cli("init").returncode, 0)
        self.assertTrue(self.cfg.exists())
        self.assertEqual(self.cli("init").returncode, 1)      # exists: leaves it alone

    def test_free_dry_run_lists_what_each_preset_does(self):
        cfg = {"residents": {"3090": [{"type": "ollama"}, {"type": "vllm"}, "whisper",
                                      {"name": "comfy", "free": "true"}]}}
        r = self.cli("free", "3090", "--dry-run", cfg=cfg)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ollama: unload models over HTTP", r.stdout)
        self.assertIn("vllm: stop vllm.service", r.stdout)
        self.assertIn("whisper: no way to free it declared", r.stdout)
        self.assertIn("comfy: run true", r.stdout)


class Free(unittest.TestCase):
    """free_card / restore in-process, with systemctl replaced and a fake Ollama on localhost."""

    def setUp(self):
        self.calls = []
        self._sc, self._state = g._systemctl, g.STATE
        g._systemctl = lambda *a: (self.calls.append(a), subprocess.CompletedProcess(a, 0, "", ""))[1]
        g.STATE = Path(tempfile.mkdtemp(prefix="gpuq-free-"))

    def tearDown(self):
        g._systemctl, g.STATE = self._sc, self._state

    def test_stop_records_and_restore_starts(self):
        cfg = {"residents": {"3090": [{"type": "vllm"}]}}
        rc, stopped = g.free_card("3090", cfg, say=lambda t: None)
        self.assertEqual((rc, stopped), (0, ["vllm.service"]))
        self.assertEqual(g.read_freed(), {"vllm.service": "3090"})
        self.assertIn(("stop", "vllm.service"), self.calls)
        args = type("A", (), {"card": None, "dry_run": False})()
        old = g.config
        g.config = lambda: cfg
        try:
            with open(os.devnull, "w") as dn, unittest.mock.patch("sys.stdout", dn):
                self.assertEqual(g._restore(args, T0), 0)
        finally:
            g.config = old
        self.assertIn(("start", "vllm.service"), self.calls)
        self.assertEqual(g.read_freed(), {})

    def test_ollama_unload_over_http(self):
        seen = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"models": [{"name": "qwen3:8b", "size_vram": 1}]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = "http://127.0.0.1:%d" % srv.server_port
            cfg = {"residents": {"3060": [{"type": "ollama", "url": url}]}}
            rc, stopped = g.free_card("3060", cfg, say=lambda t: None)
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual((rc, stopped), (0, []))
        self.assertEqual(seen, [("/api/generate", {"model": "qwen3:8b", "keep_alive": 0})])

    def test_unreachable_server_is_an_error_not_a_crash(self):
        cfg = {"residents": {"3060": [{"type": "ollama", "url": "http://127.0.0.1:9"}]}}
        rc, _ = g.free_card("3060", cfg, say=lambda t: None)
        self.assertEqual(rc, 1)


class MCP(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gpuq-mcp-"))
        self.env = {"GPUQ_STATE": str(self.tmp), "GPUQ_CONFIG": str(self.tmp / "none.json"),
                    "GPUQ_CARDS": json.dumps(CARDS), "GPUQ_COMPAT_LOCK": ""}
        self.now = [T0]

        def runner(args, host=None):        # the CLI in a subprocess, on our clock and state
            env = dict(os.environ, PYTHONPATH=str(ROOT), GPUQ_NOW=str(self.now[0]), **self.env)
            r = subprocess.run([sys.executable, "-m", "gpuq", *args], capture_output=True, text=True, env=env)
            return r.returncode, r.stdout, r.stderr
        self.srv = mcp.Server(runner)

    def call(self, name, **a):
        rep = self.srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                               "params": {"name": name, "arguments": a}})
        return rep["result"]["structuredContent"]

    def test_handshake_and_tool_list(self):
        rep = self.srv.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                               "params": {"protocolVersion": "2025-03-26"}})
        self.assertEqual(rep["result"]["protocolVersion"], "2025-03-26")
        self.assertIn("tools", rep["result"]["capabilities"])
        self.assertIsNone(self.srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        names = [t["name"] for t in self.srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
                 ["result"]["tools"]]
        self.assertEqual(names[:2], ["gpuq_inventory", "gpuq_where"])
        self.assertIn("error", self.srv.handle({"jsonrpc": "2.0", "id": 3, "method": "nope"}))
        rep = self.srv.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "x"}})
        self.assertEqual(rep["error"]["code"], -32602)

    def test_join_start_and_yield_notice_on_another_call(self):
        a = self.call("gpuq_join", owner="agent-a", job="train", est_min=600, vram_gb=16)
        self.assertEqual(a["gpu"], "3090")
        st = self.call("gpuq_start", id=a["id"])
        self.assertEqual((st["state"], st["env"]), ("go", {"CUDA_VISIBLE_DEVICES": "GPU-aaaa"}))
        self.now[0] += 60
        b = self.call("gpuq_join", owner="agent-b", job="quick", est_min=5, gpu="3090")
        # agent A's session makes an unrelated call: its held job is beaten and the yield surfaces
        res = self.call("gpuq_show")
        self.assertEqual(res["lines"]["3090"][0]["id"], b["id"])
        self.assertEqual(res["notices"][0]["state"], "yield")
        self.assertEqual(self.call("gpuq_beat", id=a["id"])["state"], "yield")
        self.assertEqual(self.call("gpuq_start", id=a["id"])["state"], "wait")   # acknowledged
        self.assertEqual(self.call("gpuq_done", id=a["id"])["ok"], True)
        self.assertEqual(self.srv.held, {})

    def test_where_and_inventory(self):
        res = self.call("gpuq_where", vram_gb=10, hosts=g.HOST)
        self.assertEqual(res["result"][0]["card"], "3060")   # both free: the smaller card that fits
        inv = self.call("gpuq_inventory", hosts=g.HOST)
        self.assertIn("3060", inv["result"][g.HOST])

    def test_stdio_loop(self):
        lines = [json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}), "not json",
                 json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})]
        out = io.StringIO()
        mcp.serve(io.StringIO("\n".join(lines) + "\n"), out)
        replies = [json.loads(x) for x in out.getvalue().splitlines()]
        self.assertEqual(replies[0]["result"], {})
        self.assertEqual(replies[1]["error"]["code"], -32700)
        self.assertEqual(len(replies), 2)


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()


class Display(unittest.TestCase):
    def test_model_names(self):
        from gpuq import cli as g
        self.assertEqual(g.nvidia_model("NVIDIA GeForce RTX 3090"), "RTX 3090")
        self.assertEqual(g.amd_model("Strix [Radeon 880M / 890M]"), "Radeon 880M")

    def test_summary_row_per_gpu(self):
        from gpuq import cli as g
        snap = {"host": "a", "entries": [], "card": None, "meters": {"3090": {"used_mib": 12288, "util": 50}},
                "cards": {"3090": {"mib": 24576, "model": "RTX 3090"}}}
        (_, _), (text, st) = g.summary_lines([("a", snap, None)], ["a"])
        self.assertIn("RTX 3090", text)
        self.assertNotIn(" a ", text)
        self.assertIn("█████░░░░░", text)
        self.assertIn("idle", text)
        self.assertEqual(st.partition("|")[0], "m0")


class Priority(unittest.TestCase):
    def setUp(self):
        from gpuq import core
        self.c = core

    def line(self, *jobs):
        q = []
        for owner, est, pr in jobs:
            self.c.join(q, 0, owner, owner, est_min=est, priority=pr)
        return q

    def test_higher_goes_ahead_equals_keep_order(self):
        q = self.line(("a", 600, 10), ("b", 600, 10), ("chill", 600, 5), ("imp", 600, 20), ("c", 600, 10))
        self.assertEqual([e["owner"] for e in q], ["imp", "a", "b", "c", "chill"])

    def test_outranking_the_runner_makes_it_yield(self):
        q = self.line(("a", 600, 10))
        self.c.start(q, 0, q[0]["id"])
        self.c.join(q, 60, "ablaze", "u", est_min=600, priority=30)
        self.assertEqual(q[0]["owner"], "ablaze")
        self.assertEqual(q[1]["state"], "yielding")

    def test_equal_priority_does_not_preempt(self):
        q = self.line(("a", 600, 20))
        self.c.start(q, 0, q[0]["id"])
        self.c.join(q, 60, "b", "b", est_min=600, priority=20)
        self.assertEqual([e["owner"] for e in q], ["a", "b"])
        self.assertEqual(q[0]["state"], "running")

    def test_short_job_cannot_jump_higher_priority(self):
        q = self.line(("big", 6000, 15))
        self.c.join(q, 0, "short", "s", est_min=5, priority=10)
        self.c.jump(q, 0, q[-1]["id"])
        self.assertEqual(q[0]["owner"], "big")

    def test_old_entries_read_as_important_or_normal(self):
        self.assertEqual(self.c.prio({"important": True}), 20)
        self.assertEqual(self.c.prio({}), 10)
        self.assertEqual(self.c.prio_mark({"priority": 5}), "☾")
        self.assertEqual(self.c.prio_mark({"priority": 30}), "⚑")

    def test_preempt_flag_is_one_above_runner(self):
        from gpuq import cli
        q = self.line(("a", 600, 20))
        self.c.start(q, 0, q[0]["id"])
        ns = type("A", (), {"priority": None, "level": "preempt"})()
        self.assertEqual(cli.want_priority(ns, q), 21)


class TuiTabs(unittest.TestCase):
    def test_only_the_focused_gpu_is_listed(self):
        from gpuq import cli as g, core
        views = []
        for h in ("a", "b"):
            q = []
            core.join(q, 0, "own-" + h, "job-" + h)
            views.append((h, {"host": h, "entries": q, "now": 0, "cards": {}, "history": []}, None))
        text = "\n".join(t for t, _ in g.tui_lines(views, 1, 0, 0, 100, only_focus=True))
        self.assertIn("own-b", text)
        self.assertNotIn("own-a  —", text)
        rows = [(t, st) for t, st in g.tui_lines(views, 1, 0, 0, 100, only_focus=True) if st.startswith("m")]
        self.assertTrue(rows[1][0].startswith("▸") and rows[1][1].partition("|")[0].endswith("s"))
        self.assertFalse(rows[0][1].partition("|")[0].endswith("s"))
