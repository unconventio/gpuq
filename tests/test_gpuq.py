"""Tests for gpuq — the per-host GPU queue .

Pure queue functions are tested with an injected clock; the CLI is tested by running
the script with GPUQ_STATE / GPUQ_COMPAT_LOCK / GPUQ_NOW pointed at a temp dir.
Run from the repo root: python3 -m unittest discover -s tests -v
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Module-level paths are read at import: point them at a temp dir first, so nothing in these
# tests can ever touch a real queue.
os.environ["GPUQ_STATE"] = tempfile.mkdtemp(prefix="gpuq-test-")
os.environ["GPUQ_CONFIG"] = os.path.join(os.environ["GPUQ_STATE"], "no-config.json")
os.environ["GPUQ_COMPAT_LOCK"] = ""
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gpuq import cli as g  # noqa: E402

T0 = 1_760_000_000.0
MIN = 60


def queue(*owners, now=T0):
    q = []
    for o in owners:
        g.join(q, now, o, "job-" + o, models=["m"], est_min=60)
    return q


class Core(unittest.TestCase):
    def test_join_appends_waiting_and_marks_only_head(self):
        q = queue("a", "b")
        self.assertEqual([e["state"] for e in q], ["waiting", "waiting"])
        self.assertEqual(q[0]["head_since"], T0)
        self.assertIsNone(q[1]["head_since"])
        self.assertTrue(any(c.isalpha() for c in q[0]["id"]))

    def test_start_at_head_runs(self):
        q = queue("a", "b")
        e = g.start(q, T0 + 5, q[0]["id"])
        self.assertEqual(e["state"], "running")
        self.assertEqual(e["heartbeat"], T0 + 5)
        self.assertIsNone(e["head_since"])

    def test_start_not_head_refused(self):
        q = queue("a", "b")
        with self.assertRaises(g.QueueError) as cm:
            g.start(q, T0, q[1]["id"])
        self.assertEqual(cm.exception.code, 3)

    def test_beat_unknown_id(self):
        q = queue("a")
        with self.assertRaises(g.QueueError) as cm:
            g.beat(q, T0, "zz99")
        self.assertEqual(cm.exception.code, 4)

    def test_beat_by_position_is_not_accepted(self):
        q = queue("a")
        with self.assertRaises(g.QueueError):
            g.beat(q, T0, "1")

    def test_beat_updates_progress(self):
        q = queue("a")
        g.start(q, T0, q[0]["id"])
        g.beat(q, T0 + 60, q[0]["id"], "41/80")
        self.assertEqual(q[0]["heartbeat"], T0 + 60)
        self.assertEqual(q[0]["progress"], "41/80")

    def test_done_promotes_next(self):
        q = queue("a", "b")
        g.start(q, T0, q[0]["id"])
        g.done(q, T0 + 100, q[0]["id"])
        self.assertEqual(q[0]["owner"], "b")
        self.assertEqual(q[0]["head_since"], T0 + 100)

    def test_move_reorders(self):
        q = queue("a", "b", "c")
        g.move(q, T0, "3", "1")
        self.assertEqual([e["owner"] for e in q], ["c", "a", "b"])
        self.assertEqual(q[0]["head_since"], T0)
        self.assertIsNone(q[1]["head_since"])

    def test_move_above_running_asks_it_to_yield(self):
        q = queue("a", "b", "c")
        g.start(q, T0, q[0]["id"])
        g.move(q, T0 + 5, "3", "1")
        self.assertEqual([(e["owner"], e["state"]) for e in q],
                         [("c", "waiting"), ("a", "yielding"), ("b", "waiting")])
        self.assertEqual(q[1]["yield_since"], T0 + 5)

    def test_move_running_down_asks_it_to_yield(self):
        q = queue("a", "b")
        g.start(q, T0, q[0]["id"])
        g.move(q, T0, "1", "2")
        self.assertEqual([(e["owner"], e["state"]) for e in q], [("b", "waiting"), ("a", "yielding")])

    def test_moving_yielder_back_to_head_calls_it_off(self):
        q = queue("a", "b")
        g.start(q, T0, q[0]["id"])
        g.move(q, T0, "1", "2")
        g.move(q, T0, "2", "1")
        self.assertEqual([(e["owner"], e["state"]) for e in q], [("a", "running"), ("b", "waiting")])

    def test_removing_the_head_above_a_yielder_calls_it_off(self):
        q = queue("a", "b")
        g.start(q, T0, q[0]["id"])
        g.move(q, T0, "1", "2")
        g.remove(q, T0, "1")
        self.assertEqual(q[0]["state"], "running")

    def test_yielder_beat_exits_5_and_saves_heartbeat(self):
        q = queue("a", "b")
        g.start(q, T0, q[0]["id"])
        g.move(q, T0, "1", "2")
        with self.assertRaises(g.QueueError) as cm:
            g.beat(q, T0 + 60, q[1]["id"], "5/9")
        self.assertEqual(cm.exception.code, 5)
        self.assertTrue(cm.exception.save)
        self.assertEqual((q[1]["heartbeat"], q[1]["progress"]), (T0 + 60, "5/9"))

    def test_head_cannot_start_until_yielder_stops(self):
        q = queue("a", "b")
        a = q[0]["id"]
        g.start(q, T0, a)
        g.move(q, T0, "1", "2")
        with self.assertRaises(g.QueueError) as cm:
            g.start(q, T0 + 30, q[0]["id"])
        self.assertEqual(cm.exception.code, 3)
        with self.assertRaises(g.QueueError) as cm:   # the yielder's start loop acknowledges
            g.start(q, T0 + 60, a)
        self.assertEqual((cm.exception.code, cm.exception.save), (3, True))
        self.assertEqual(q[1]["state"], "waiting")
        self.assertEqual(g.start(q, T0 + 90, q[0]["id"])["state"], "running")
        g.done(q, T0 + 200, q[0]["id"])
        self.assertEqual(g.start(q, T0 + 210, a)["state"], "running")   # handed back

    def test_sweep_holds_head_clock_while_yielding(self):
        q = queue("a", "b")
        g.start(q, T0, q[0]["id"])
        g.move(q, T0, "1", "2")
        events = g.sweep(q, T0 + 14 * MIN) + g.sweep(q, T0 + 14 * MIN + 1)
        self.assertEqual(events, [])
        self.assertEqual(q[0]["head_since"], T0 + 14 * MIN + 1)

    def test_sweep_purges_silent_yielder(self):
        q = queue("a", "b")
        g.start(q, T0, q[0]["id"])
        g.move(q, T0, "1", "2")
        events = g.sweep(q, T0 + 16 * MIN)
        self.assertEqual([(ev, e["owner"]) for ev, e, _ in events], [("purged", "a")])
        self.assertEqual([e["owner"] for e in q], ["b"])

    def _q(self, *jobs):
        """jobs: (owner, est_min[, important]); the first one starts."""
        q = []
        for j in jobs:
            g.join(q, T0, j[0], "job-" + j[0], est_min=j[1], important=len(j) > 2 and j[2])
        g.start(q, T0, q[0]["id"])
        return q

    def test_short_job_jumps_long_runner_which_yields(self):
        q = self._q(("carol", 3000), ("art", 60))
        passed = g.jump(q, T0 + 60, q[1]["id"])
        self.assertEqual([x["owner"] for x in passed], ["carol"])
        self.assertEqual([(e["owner"], e["state"]) for e in q],
                         [("art", "waiting"), ("carol", "yielding")])

    def test_jump_stops_at_a_peer_short_job(self):
        q = self._q(("carol", 3000), ("lora", 190), ("test", 8))
        g.jump(q, T0, q[1]["id"])
        g.jump(q, T0, q[2]["id"])
        # test (8m) passes carol but not lora? lora has 190m >= 10*8m, so test passes lora too
        self.assertEqual([e["owner"] for e in q], ["test", "lora", "carol"])

    def test_jump_keeps_order_among_similar_jobs(self):
        q = self._q(("carol", 3000), ("a", 30), ("b", 20))
        g.jump(q, T0, q[1]["id"])
        g.jump(q, T0, q[2]["id"])
        self.assertEqual([e["owner"] for e in q], ["a", "b", "carol"])

    def test_important_job_is_not_jumped(self):
        q = self._q(("carol", 3000, True), ("art", 60))
        self.assertEqual(g.jump(q, T0, q[1]["id"]), [])
        self.assertEqual(q[0]["state"], "running")

    def test_no_estimate_or_too_long_does_not_jump(self):
        q = self._q(("carol", 6000), ("x", None), ("y", 7 * 60))
        self.assertEqual(g.jump(q, T0, q[1]["id"]), [])
        self.assertEqual(g.jump(q, T0, q[2]["id"]), [])

    def test_runner_remaining_counts_elapsed(self):
        q = self._q(("carol", 600), ("art", 50))
        self.assertEqual(g.jump(q, T0 + 200 * MIN, q[1]["id"]), [])   # 400m left < 10*50m

    def test_move_out_of_range(self):
        q = queue("a", "b")
        with self.assertRaises(g.QueueError) as cm:
            g.move(q, T0, "1", "5")
        self.assertEqual(cm.exception.code, 2)

    def test_remove_by_position(self):
        q = queue("a", "b")
        e = g.remove(q, T0, "2")
        self.assertEqual(e["owner"], "b")
        self.assertEqual(len(q), 1)


class Sweep(unittest.TestCase):
    def test_stale_holder_purged_and_next_promoted(self):
        q = queue("a", "b")
        g.start(q, T0, q[0]["id"])
        now = T0 + 16 * MIN
        ev = g.sweep(q, now)
        self.assertEqual([(k, e["owner"]) for k, e, _ in ev], [("purged", "a")])
        self.assertEqual(q[0]["owner"], "b")
        self.assertEqual(q[0]["head_since"], now)

    def test_fresh_holder_kept(self):
        q = queue("a")
        g.start(q, T0, q[0]["id"])
        self.assertEqual(g.sweep(q, T0 + 14 * MIN), [])
        self.assertEqual(q[0]["state"], "running")

    def test_overrun_purged(self):
        q = []
        g.join(q, T0, "a", "j", max_min=30)
        g.start(q, T0, q[0]["id"])
        g.beat(q, T0 + 31 * MIN, q[0]["id"])
        ev = g.sweep(q, T0 + 31 * MIN)
        self.assertEqual(ev[0][0], "purged")
        self.assertEqual(q, [])

    def test_noshow_skipped_then_removed(self):
        q = queue("a", "b")
        ev = g.sweep(q, T0 + 16 * MIN)
        self.assertEqual([(k, e["owner"]) for k, e, _ in ev], [("skipped", "a")])
        self.assertEqual([e["owner"] for e in q], ["b", "a"])
        self.assertEqual(q[1]["skips"], 1)
        self.assertIsNone(q[1]["head_since"])
        g.remove(q, T0 + 17 * MIN, "1")           # b leaves; a is #1 again
        self.assertEqual(g.sweep(q, T0 + 20 * MIN), [])
        ev = g.sweep(q, T0 + 33 * MIN)
        self.assertEqual([(k, e["owner"]) for k, e, _ in ev], [("removed", "a")])
        self.assertEqual(q, [])

    def test_lone_noshow_gets_a_second_grace(self):
        q = queue("a")
        self.assertEqual(g.sweep(q, T0 + 16 * MIN)[0][0], "skipped")
        self.assertEqual(g.sweep(q, T0 + 17 * MIN), [])
        self.assertEqual(g.sweep(q, T0 + 32 * MIN)[0][0], "removed")

    def test_start_resets_skips(self):
        q = queue("a")
        g.sweep(q, T0 + 16 * MIN)
        g.start(q, T0 + 17 * MIN, q[0]["id"])
        self.assertEqual(q[0]["skips"], 0)


class Pause(unittest.TestCase):
    def test_paused_sweep_does_not_skip_and_resume_gives_fresh_grace(self):
        q = queue("a", "b")
        self.assertEqual(g.sweep(q, T0 + 60 * MIN, paused=True), [])
        self.assertEqual(q[0]["owner"], "a")
        self.assertEqual(q[0]["head_since"], T0 + 60 * MIN)   # clock held at "now" while paused
        self.assertEqual(g.sweep(q, T0 + 70 * MIN), [])        # 10 min after resume: still fine
        self.assertEqual(g.sweep(q, T0 + 76 * MIN)[0][0], "skipped")

    def test_paused_still_purges_a_stale_runner(self):
        q = queue("a")
        g.start(q, T0, q[0]["id"])
        self.assertEqual(g.sweep(q, T0 + 16 * MIN, paused=True)[0][0], "purged")

    def test_render_shows_paused(self):
        g.COMPAT_LOCK = ""
        snap = g.snapshot(queue("a"), T0, [], None, host="hostB", paused={"since": T0})
        self.assertIn("PAUSED", g.render(snap))
        self.assertIn("PAUSED", "\n".join(t for t, _ in g.tui_lines([("hostB", snap, None)], 0, 0, T0)))


class View(unittest.TestCase):
    def setUp(self):
        g.COMPAT_LOCK = ""   # never read a real lock file in tests

    def test_render_running_and_eta(self):
        q = queue("alice", "bob")
        g.start(q, T0, q[0]["id"])
        snap = g.snapshot(q, T0 + 30 * MIN, [], ["gpt-oss:20b"], host="hostA")
        out = g.render(snap)
        self.assertIn("hostA", out)
        self.assertIn(" 1 ", out)
        self.assertIn("alice", out)
        self.assertIn("RUNNING", out)
        self.assertIn("starts ~", out)
        self.assertNotIn("unqueued", out)

    def test_render_flags_unqueued_gpu_use(self):
        snap = g.snapshot([], T0, [], ["qwen3:8b"], host="hostA")
        self.assertIn("unqueued GPU use", g.render(snap))


class TuiLayout(unittest.TestCase):
    def setUp(self):
        g.COMPAT_LOCK = ""

    def views(self):
        q = queue("alice", "bob")
        g.start(q, T0, q[0]["id"])
        q[1]["why"] = "image gen for the wall"
        snap = g.snapshot(q, T0, [{"t": T0, "event": "joined", "owner": "bob",
                                   "job": "job-bob", "note": ""}], ["m"], host="hostA")
        return [("hostA", snap, None), ("hostB", None, "hostB: unreachable (timeout)")]

    def test_rows_and_selection(self):
        lines = g.tui_lines(self.views(), 0, 1, T0 + 60)
        rows = [(t, st) for t, st in lines if re.match(r"^\s*\d+ [▶ ] ", t)]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][1], "run")
        self.assertEqual(rows[1][1], "sel")
        self.assertIn("bob", rows[1][0])

    def test_detail_shows_selected_why_and_id(self):
        v = self.views()
        lines = "\n".join(t for t, _ in g.tui_lines(v, 0, 1, T0 + 60))
        self.assertIn("image gen for the wall", lines)
        self.assertIn(v[0][1]["entries"][1]["id"], lines)

    def test_active_host_is_marked(self):
        v = self.views()
        v[1] = ("hostB", g.snapshot([], T0, [], None, host="hostB"), None)
        heads = [(t, st) for t, st in g.tui_lines(v, 1, 0, T0)
                 if st.startswith("h") and st[1:2].isdigit()]
        self.assertTrue(heads[0][0].startswith("  hostA") and not heads[0][1].endswith("s"))
        self.assertTrue(heads[1][0].startswith("▸ hostB") and heads[1][1].endswith("s"))

    def test_unreachable_host_is_a_header_not_a_crash(self):
        lines = g.tui_lines(self.views(), 1, 0, T0)
        self.assertTrue(any("unreachable" in t and st.startswith("h") and st[1:2].isdigit()
                            for t, st in lines))


class Evac(unittest.TestCase):
    def test_unit_for_uses_config_then_default(self):
        cfg = {"units": {"voice-app": "voice.service"}}
        self.assertEqual(g._unit_for("voice-app", cfg), "voice.service")
        self.assertEqual(g._unit_for("whisper-server", cfg), "whisper-server.service")

    def test_evac_targets_defaults_and_uuid(self):
        cards = {"3060": {"uuid": "GPU-60", "mib": 12288}, "3090": {"uuid": "GPU-90", "mib": 24576}}
        cfg = {"default": "3090", "residents": {"3090": ["voice-app", "whisper-server"]},
               "units": {"voice-app": "voice.service"}}
        card, to, uuid, units = g.evac_targets(cards, cfg)
        self.assertEqual((card, to, uuid), ("3090", "3060", "GPU-60"))
        self.assertEqual(units, ["voice.service", "whisper-server.service"])

    def test_evac_targets_needs_a_second_card(self):
        with self.assertRaises(g.QueueError):
            g.evac_targets({"3090": {"uuid": "u", "mib": 1}},
                           {"default": "3090", "residents": {"3090": ["x"]}})

    def test_evac_targets_unknown_card(self):
        with self.assertRaises(g.QueueError):
            g.evac_targets({"3090": {"uuid": "u", "mib": 1}}, {}, card="4090")

    def test_dropin_path(self):
        self.assertTrue(str(g._evac_dropin("voice.service"))
                        .endswith("systemd/user/voice.service.d/gpuq-evac.conf"))

    def test_header_shows_evacuated_on_that_card(self):
        g.COMPAT_LOCK = ""
        snap = g.snapshot([], T0, [], None, host="hostB", cards={"3090": {"uuid": "u", "mib": 24576}},
                          default="3090", evac={"card": "3090", "to": "3060"})
        snap["card"] = "3090"
        self.assertIn("EVACUATED → 3060", g.header(snap, T0))


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.lock = self.dir / "LOCK.txt"

    def tearDown(self):
        self.tmp.cleanup()

    cards = {"gpu": {"uuid": "GPU-one", "mib": 8192}}

    def run_cli(self, *args, now=T0):
        env = dict(os.environ, GPUQ_STATE=str(self.dir / "state"),
                   GPUQ_COMPAT_LOCK=str(self.lock), GPUQ_NOW=str(now),
                   GPUQ_CARDS=json.dumps(self.cards), GPUQ_CONFIG=str(self.dir / "config.json"))
        return subprocess.run([sys.executable, "-m", "gpuq", *args],
                              capture_output=True, text=True, env=env)

    def join(self, owner):
        r = self.run_cli("join", "--owner", owner, "--job", "j", "--est", "60", "--max", "600")
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def test_start_writes_compat_lock_and_done_clears_it(self):
        eid = self.join("alice")
        self.assertEqual(self.run_cli("start", eid).returncode, 0)
        self.assertIn("gpuq=%s" % eid, self.lock.read_text())
        self.assertEqual(self.run_cli("done", eid).returncode, 0)
        self.assertFalse(self.lock.exists())

    def test_foreign_lock_refuses_start_unless_adopt(self):
        self.lock.write_text("owner=someone\njob=x\n")
        eid = self.join("alice")
        self.assertEqual(self.run_cli("start", eid).returncode, 3)
        self.assertEqual(self.run_cli("start", "--adopt", eid).returncode, 0)
        self.assertIn("gpuq=%s" % eid, self.lock.read_text())

    def test_done_leaves_foreign_lock(self):
        eid = self.join("alice")
        self.run_cli("start", eid)
        self.lock.write_text("owner=someone-else\n")
        self.run_cli("done", eid)
        self.assertTrue(self.lock.exists())

    def test_sweep_purges_logs_and_clears_lock(self):
        eid = self.join("alice")
        self.run_cli("start", eid)
        r = self.run_cli("sweep", now=T0 + 16 * MIN)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(self.lock.exists())
        hist = (self.dir / "state" / "history.jsonl").read_text()
        self.assertIn('"purged"', hist)

    def test_yield_round_trip(self):
        cap, aur = self.join("carol"), self.join("bob")
        self.run_cli("start", cap)
        self.assertEqual(self.run_cli("yield", cap, now=T0 + 10).returncode, 0)
        r = self.run_cli("beat", cap, now=T0 + 20)
        self.assertEqual(r.returncode, 5, r.stderr)
        self.assertEqual(self.run_cli("start", aur, now=T0 + 25).returncode, 3)
        self.assertEqual(self.run_cli("start", cap, now=T0 + 30).returncode, 3)   # ack
        self.assertFalse(self.lock.exists())
        self.assertEqual(self.run_cli("start", aur, now=T0 + 40).returncode, 0)
        self.assertEqual(self.run_cli("start", cap, now=T0 + 50).returncode, 3)
        self.run_cli("done", aur, now=T0 + 60)
        self.assertEqual(self.run_cli("start", cap, now=T0 + 70).returncode, 0)
        hist = (self.dir / "state" / "history.jsonl").read_text()
        self.assertIn('"yielded"', hist)

    def test_join_short_job_jumps_and_reports_on_stderr(self):
        cap = self.run_cli("join", "--owner", "carol", "--job", "c", "--est", "3000").stdout.strip()
        self.run_cli("start", cap)
        r = self.run_cli("join", "--owner", "art", "--job", "a", "--est", "30", now=T0 + 5)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertRegex(r.stdout.strip(), r"^[0-9a-f]{4}$")
        self.assertIn("jumped to #1", r.stderr)
        self.assertEqual(self.run_cli("beat", cap, now=T0 + 10).returncode, 5)

    def test_edit_important_then_not(self):
        cap = self.run_cli("join", "--owner", "carol", "--job", "c", "--est", "3000",
                           "--important").stdout.strip()
        self.run_cli("start", cap)
        art = self.run_cli("join", "--owner", "art", "--job", "a", "--est", "30").stdout.strip()
        self.assertEqual(self.run_cli("start", art).returncode, 3)
        self.run_cli("edit", art, "--est", "20")
        self.assertEqual(self.run_cli("beat", cap).returncode, 0)
        self.run_cli("edit", cap, "--normal")
        self.run_cli("edit", art, "--est", "15")
        self.assertEqual(self.run_cli("beat", cap).returncode, 5)

    def wait_for(self, cond, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return True
            time.sleep(0.05)
        self.fail("timed out waiting")

    def state(self, eid):
        lines = json.loads((self.dir / "state" / "queue.json").read_text())["lines"]
        return next((e["state"] for q in lines.values() for e in q if e["id"] == eid), None)

    def test_run_yields_reruns_and_releases(self):
        cap, art = self.join("carol"), self.join("art")
        runs = self.dir / "runs"
        # first run hangs until stopped; the rerun finishes
        job = ("echo x >> %s; [ $(wc -l < %s) -ge 2 ] && exit 0; sleep 60" % (runs, runs))
        env = dict(os.environ, GPUQ_STATE=str(self.dir / "state"),
                   GPUQ_COMPAT_LOCK=str(self.lock), GPUQ_NOW=str(T0))
        w = subprocess.Popen([sys.executable, "-m", "gpuq", "run", cap, "--every", "0.2",
                              "--on-stop", "echo stopped >> %s" % (self.dir / "stops"),
                              "--", "sh", "-c", job], env=env, stderr=subprocess.PIPE, text=True)
        try:
            self.wait_for(lambda: self.state(cap) == "running" and runs.exists())
            self.assertEqual(self.run_cli("yield", cap).returncode, 0)
            self.wait_for(lambda: self.state(cap) == "waiting")
            self.assertEqual((self.dir / "stops").read_text(), "stopped\n")
            self.assertEqual(self.run_cli("start", art).returncode, 0)
            self.run_cli("done", art)
            self.assertEqual(w.wait(timeout=10), 0)
        finally:
            if w.poll() is None:
                w.kill()
        self.assertEqual(runs.read_text(), "x\nx\n")
        self.assertIsNone(self.state(cap))      # done: left the line

    def test_yield_alone_refused(self):
        cap = self.join("carol")
        self.run_cli("start", cap)
        self.assertEqual(self.run_cli("yield", cap).returncode, 2)

    def test_beat_after_purge_exits_4(self):
        eid = self.join("alice")
        self.run_cli("start", eid)
        self.run_cli("sweep", now=T0 + 16 * MIN)
        self.assertEqual(self.run_cli("beat", eid, now=T0 + 17 * MIN).returncode, 4)

    def test_bare_command_shows(self):
        self.join("a")
        r = self.run_cli()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("waiting", r.stdout)

    def test_act_runs_local_commands(self):
        env = {"GPUQ_STATE": str(self.dir / "state"), "GPUQ_COMPAT_LOCK": str(self.lock),
               "GPUQ_NOW": str(T0)}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            a, b = self.join("a"), self.join("b")
            self.assertEqual(g.act(g.HOST, "mv", b, "1")[0], 0)
            code, msg = g.act(g.HOST, "start", a)
            self.assertEqual(code, 3)
            self.assertIn("not #1", msg)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_pause_blocks_start_until_resume(self):
        eid = self.join("a")
        self.assertEqual(self.run_cli("pause").returncode, 0)
        r = self.run_cli("start", eid)
        self.assertEqual(r.returncode, 3)
        self.assertIn("paused", r.stderr)
        self.assertTrue(json.loads(self.run_cli("show", "--json").stdout)["paused"])
        self.assertEqual(self.run_cli("resume").returncode, 0)
        self.assertEqual(self.run_cli("start", eid).returncode, 0)

    def test_edit_updates_fields_keeps_id_and_place(self):
        a, b = self.join("a"), self.join("b")
        r = self.run_cli("edit", b, "--est", "120", "--max", "180", "--models", "x,y", "--why", "w")
        self.assertEqual(r.returncode, 0, r.stderr)
        e = json.loads(self.run_cli("show", "--json").stdout)["entries"][1]
        self.assertEqual((e["id"], e["est_min"], e["max_min"], e["models"], e["why"]),
                         (b, 120, 180, ["x", "y"], "w"))

    def test_show_json_order_and_mv(self):
        a, b = self.join("a"), self.join("b")
        self.assertEqual(self.run_cli("mv", b, "1").returncode, 0)
        r = self.run_cli("show", "--json")
        snap = json.loads(r.stdout)
        self.assertEqual([e["id"] for e in snap["entries"]], [b, a])



class TwoCards(Cli):
    """hostB since 2026-10-28: a line per card; the 3090 is the default (most memory)."""
    cards = {"3060": {"uuid": "GPU-small", "mib": 12288}, "3090": {"uuid": "GPU-big", "mib": 24576}}

    def lines(self):
        return json.loads((self.dir / "state" / "queue.json").read_text())["lines"]

    def join_on(self, owner, gpu=None):
        r = self.run_cli("join", "--owner", owner, "--job", "j", "--est", "60", "--max", "600",
                         *(["--gpu", gpu] if gpu else []))
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def test_default_is_the_biggest_card_and_lines_are_separate(self):
        a, b = self.join_on("a"), self.join_on("b", "3060")
        self.assertEqual([e["id"] for e in self.lines()["3090"]], [a])
        self.assertEqual([e["id"] for e in self.lines()["3060"]], [b])
        # both are #1 in their own line: both may run at once
        ra, rb = self.run_cli("start", a), self.run_cli("start", b)
        self.assertEqual((ra.returncode, rb.returncode), (0, 0), ra.stderr + rb.stderr)
        self.assertEqual(ra.stdout.strip(), "CUDA_VISIBLE_DEVICES=GPU-big")
        self.assertEqual(rb.stdout.strip(), "CUDA_VISIBLE_DEVICES=GPU-small")
        self.assertEqual(self.run_cli("env", b).stdout.strip(), "CUDA_VISIBLE_DEVICES=GPU-small")

    def test_config_default_and_owner_default(self):
        (self.dir / "config.json").write_text(json.dumps({"default": "3060", "owners": {"alice": "3090"}}))
        a, z = self.join_on("a"), self.join_on("alice")
        self.assertEqual([e["id"] for e in self.lines()["3060"]], [a])
        self.assertEqual([e["id"] for e in self.lines()["3090"]], [z])
        self.assertEqual(json.loads(self.run_cli("show", "--json").stdout)["entries"][0]["id"], a)

    def test_positions_are_per_line_ids_work_anywhere(self):
        a, b = self.join_on("a", "3060"), self.join_on("b", "3060")
        c = self.join_on("c")
        self.assertEqual(self.run_cli("mv", "2", "1", "--gpu", "3060").returncode, 0)
        self.assertEqual([e["id"] for e in self.lines()["3060"]], [b, a])
        self.assertEqual(self.run_cli("rm", a).returncode, 0)
        self.assertEqual(self.run_cli("rm", "1").returncode, 0)   # default line: c
        self.assertEqual(self.lines()["3090"], [])
        self.assertEqual([e["id"] for e in self.lines()["3060"]], [b])
        self.assertNotEqual(self.run_cli("join", "--owner", "x", "--job", "j", "--gpu", "4090").returncode, 0)

    def test_ids_unique_across_lines(self):
        ids = [self.join_on("o%d" % i, ("3060", "3090")[i % 2]) for i in range(40)]
        self.assertEqual(len(set(ids)), 40)

    def test_edit_moves_a_waiting_entry_to_another_card(self):
        a, b = self.join_on("a"), self.join_on("b")
        self.assertEqual(self.run_cli("edit", b, "--gpu", "3060").returncode, 0)
        self.assertEqual([e["id"] for e in self.lines()["3060"]], [b])
        self.run_cli("start", a)
        self.assertNotEqual(self.run_cli("edit", a, "--gpu", "3060").returncode, 0)

    def test_old_single_list_file_becomes_the_default_line(self):
        state = self.dir / "state"
        state.mkdir()
        q = []
        g.join(q, T0, "old", "j")
        (state / "queue.json").write_text(json.dumps(q))
        self.assertEqual(self.run_cli("start", q[0]["id"]).returncode, 0)
        self.assertEqual(self.lines()["3090"][0]["state"], "running")
        self.assertEqual(self.lines()["3060"], [])

    def test_sweep_and_resume_cover_every_line(self):
        a = self.join_on("a", "3060")
        self.run_cli("start", a)
        r = self.run_cli("sweep", now=T0 + 20 * MIN)
        self.assertIn("purged", r.stdout)
        self.assertEqual(self.lines()["3060"], [])

    def test_show_has_a_block_per_card(self):
        self.join_on("a", "3060")
        out = self.run_cli("show").stdout
        self.assertIn("3090 (24 GB)", out)
        self.assertIn("3060 (12 GB)", out)

    def test_run_hands_the_job_its_card(self):
        a = self.join_on("a", "3060")
        out = self.dir / "env.txt"
        r = self.run_cli("run", a, "--every", "0.1", "--", "sh", "-c", "echo $CUDA_VISIBLE_DEVICES > %s" % out)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(out.read_text().strip(), "GPU-small")


class Residents(unittest.TestCase):
    def test_per_card_views_and_wrong_card_warning(self):
        g.COMPAT_LOCK = ""
        q90, q60 = queue("alice"), []
        use = {"3090": {"jobs": [], "residents": [], "misplaced": ["voice-app is on the 3090, meant for the 3060"]},
               "3060": {"jobs": ["ollama 6.0G"], "residents": ["whisper-server 0.5G"], "misplaced": []}}
        snap = g.snapshot(q90, T0, [], [], host="hostB", lines={"3090": q90, "3060": q60},
                          cards={"3090": {"mib": 24576}, "3060": {"mib": 12288}}, card_use=use)
        views = g.per_card(snap)
        self.assertEqual([v["card"] for v in views], ["3090", "3060"])
        out = g.render(snap)
        self.assertIn("wrong card: voice-app", out)
        self.assertIn("residents: whisper-server", out)
        self.assertIn("unqueued GPU use", out)          # the 3060's ollama, nothing queued there
        tui = g.tui_lines([("hostB", v, None) for v in views], 1, 0, T0)
        self.assertTrue(any("hostB · 3060" in t for t, _ in tui))


if __name__ == "__main__":
    unittest.main()
