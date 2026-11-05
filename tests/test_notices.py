"""Notices: a job that dies, is purged or stalls tells its contact; gpuq run saves a log and
reports the exit code; beats carry the card's memory and load."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("GPUQ_STATE", tempfile.mkdtemp(prefix="gpuq-test-"))
os.environ["GPUQ_DESKTOP"] = "0"        # never pop a real desktop notice from the tests
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gpuq import core  # noqa: E402

T0 = 1_760_000_000.0
MIN = 60


def entry(**kw):
    q = []
    e = core.join(q, T0, "painter", "lora", est_min=kw.pop("est_min", None))
    core.start(q, T0, e["id"])
    e.update(kw)
    return q, e


class Ending(unittest.TestCase):
    def test_nonzero_exit_fails_and_is_early_within_ten_minutes(self):
        _, e = entry()
        end = core.ending(e, T0 + 2 * MIN, 1)
        self.assertEqual((end["kind"], end["early"]), ("failed", True))
        end = core.ending(e, T0 + 11 * MIN, 1)
        self.assertEqual((end["kind"], end["early"]), ("failed", False))

    def test_quick_end_without_exit_code_is_early(self):
        # the 8f1c case: started, plain `done` 60 s later, no exit code
        _, e = entry()
        end = core.ending(e, T0 + MIN)
        self.assertEqual((end["kind"], end["early"]), ("ended early", True))

    def test_short_job_meant_to_be_short_is_fine(self):
        _, e = entry(est_min=2)
        self.assertIsNone(core.ending(e, T0 + 90)["kind"])
        self.assertIsNone(core.ending(e, T0 + 3 * MIN, 0)["kind"])

    def test_long_success_is_quiet(self):
        _, e = entry()
        self.assertIsNone(core.ending(e, T0 + 60 * MIN)["kind"])

    def test_signal_counts_as_failure(self):
        _, e = entry()
        self.assertEqual(core.ending(e, T0 + 30 * MIN, -9)["kind"], "failed")


class BeatMetrics(unittest.TestCase):
    def test_peak_and_average(self):
        q, e = entry()
        for i, (m, u) in enumerate([(4000, 50), (9000, 100), (7000, 30)]):
            core.beat(q, T0 + i * MIN, e["id"], mem_mib=m, util=u)
        self.assertEqual((e["mem_start_mib"], e["mem_peak_mib"]), (4000, 9000))
        self.assertEqual(e["util_sum"] / e["util_n"], 60)


class Stall(unittest.TestCase):
    def test_stuck_progress_reported_once(self):
        q, e = entry()
        core.beat(q, T0, e["id"], progress="3/80")
        self.assertEqual(core.stalls(q, T0 + 20 * MIN), [])
        core.beat(q, T0 + 25 * MIN, e["id"], progress="3/80")
        got = core.stalls(q, T0 + 31 * MIN)
        self.assertEqual(len(got), 1)
        self.assertIn("3/80", got[0][1])
        self.assertEqual(core.stalls(q, T0 + 40 * MIN), [])
        core.beat(q, T0 + 41 * MIN, e["id"], progress="4/80")     # it moves again: cleared
        self.assertIsNone(e["stalled"])

    def test_idle_card_reported(self):
        q, e = entry()
        core.beat(q, T0 + MIN, e["id"], util=0)
        core.beat(q, T0 + 20 * MIN, e["id"], util=0)
        self.assertEqual(core.stalls(q, T0 + 20 * MIN), [])
        got = core.stalls(q, T0 + 32 * MIN)
        self.assertEqual(len(got), 1)
        self.assertIn("idle", got[0][1])

    def test_load_resets_idle(self):
        q, e = entry()
        core.beat(q, T0 + MIN, e["id"], util=0)
        core.beat(q, T0 + 20 * MIN, e["id"], util=40)
        self.assertEqual(core.stalls(q, T0 + 40 * MIN), [])


class Cli(unittest.TestCase):
    cards = {"gpu": {"uuid": "GPU-one", "mib": 8192}}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.state = self.dir / "state"
        self.cfg = self.dir / "config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *args, now=T0):
        env = dict(os.environ, GPUQ_STATE=str(self.state), GPUQ_COMPAT_LOCK="", GPUQ_NOW=str(now),
                   GPUQ_CARDS=json.dumps(self.cards), GPUQ_CONFIG=str(self.cfg), GPUQ_DESKTOP="0")
        return subprocess.run([sys.executable, "-m", "gpuq", *args], capture_output=True, text=True,
                              env=env, cwd=str(ROOT), timeout=60)

    def join(self, *extra):
        r = self.cli("join", "--owner", "painter", "--job", "lora", *extra)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def inbox(self, *extra, now=T0):
        r = self.cli("inbox", "--json", *extra, now=now)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def history(self):
        return [json.loads(l) for l in (self.state / "history.jsonl").read_text().splitlines()]

    def test_failed_exit_notifies_the_owner_and_logs_it(self):
        jid = self.join()
        self.cli("start", jid)
        r = self.cli("done", jid, "--exit", "1", "--json", now=T0 + 2 * MIN)
        res = json.loads(r.stdout)
        self.assertEqual((res["notice"], res["early"], res["exit"]), ("failed", True, 1))
        got = self.inbox("--contact", "painter", now=T0 + 3 * MIN)
        self.assertEqual(len(got), 1)
        self.assertEqual((got[0]["kind"], got[0]["id"], got[0]["exit"]), ("died early", jid, 1))
        last = self.history()[-1]
        self.assertEqual((last["event"], last["exit"], last["early"]), ("failed", 1, True))

    def test_plain_done_soon_after_start_notifies(self):
        jid = self.join()
        self.cli("start", jid)
        self.cli("done", jid, now=T0 + MIN)
        got = self.inbox("--contact", "painter", now=T0 + 2 * MIN)
        self.assertEqual([n["kind"] for n in got], ["died early"])

    def test_success_and_quiet_are_silent(self):
        a = self.join("--est", "1")
        self.cli("start", a)
        self.cli("done", a, "--exit", "0", now=T0 + 30)
        b = self.join()
        self.cli("start", b, now=T0 + MIN)
        self.cli("done", b, "--no-notice", now=T0 + 2 * MIN)
        self.assertEqual(self.inbox(now=T0 + 3 * MIN), [])

    def test_ack_is_per_reader(self):
        jid = self.join("--contact", "painter,butler")
        self.cli("start", jid)
        self.cli("done", jid, "--exit", "2", now=T0 + MIN)
        self.assertEqual(len(self.inbox("--contact", "butler", now=T0 + 2 * MIN)), 1)
        self.assertEqual(len(self.inbox("--contact", "butler", "--ack", now=T0 + 2 * MIN)), 1)
        self.assertEqual(self.inbox("--contact", "butler", now=T0 + 3 * MIN), [])
        self.assertEqual(len(self.inbox("--contact", "painter", now=T0 + 3 * MIN)), 1)
        # the screen has its own mark: dismissing there leaves the contacts' notices alone
        self.cli("inbox", "--ack", "--reader", "screen", now=T0 + 3 * MIN)
        self.assertEqual(len(self.inbox("--contact", "painter", now=T0 + 3 * MIN)), 1)

    def test_edit_sets_contact_desktop_and_log_on_a_running_job(self):
        out = self.dir / "told.txt"
        self.cfg.write_text(json.dumps({"notify": {"command": "echo {kind} {contact} >> %s" % out}}))
        jid = self.join()
        self.cli("start", jid)
        r = self.cli("edit", jid, "--contact", "artist", "--desktop", "--log", "/tmp/x.log")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.cli("done", jid, "--exit", "1", now=T0 + MIN)
        got = self.inbox("--contact", "artist", now=T0 + 2 * MIN)
        self.assertEqual((len(got), got[0]["log"]), (1, "/tmp/x.log"))
        self.assertEqual(self.inbox("--contact", "painter", now=T0 + 2 * MIN), [])
        self.assertEqual(out.read_text().split(), ["died", "early", "artist"])

    def test_edit_clears_contact_desktop_and_log(self):
        jid = self.join("--contact", "artist", "--desktop", "--log", "/tmp/x.log")
        r = self.cli("edit", jid, "--contact", "", "--no-desktop", "--log", "")
        self.assertEqual(r.returncode, 0, r.stderr)
        q = json.loads((self.state / "queue.json").read_text())
        ent = [x for line in q["lines"].values() for x in line if x["id"] == jid][0]
        self.assertNotIn("contact", ent)
        self.assertNotIn("log", ent)
        self.assertFalse(ent.get("desktop"))

    def test_purge_notifies(self):
        jid = self.join()
        self.cli("start", jid)
        r = self.cli("sweep", now=T0 + 16 * MIN)
        self.assertIn("purged", r.stdout)
        got = self.inbox("--contact", "painter", now=T0 + 17 * MIN)
        self.assertEqual([n["kind"] for n in got], ["purged"])

    def test_stall_found_by_sweep(self):
        jid = self.join()
        self.cli("start", jid)
        self.cli("beat", jid, "--progress", "1/9", now=T0 + MIN)
        self.cli("beat", jid, "--progress", "1/9", now=T0 + 30 * MIN)
        self.cli("sweep", now=T0 + 32 * MIN)
        self.assertEqual([n["kind"] for n in self.inbox(now=T0 + 33 * MIN)], ["stalled"])
        self.cli("sweep", now=T0 + 33 * MIN)
        self.assertEqual(len(self.inbox(now=T0 + 34 * MIN)), 1)

    def test_desktop_job_runs_the_notify_command(self):
        out = self.dir / "told.txt"
        self.cfg.write_text(json.dumps({"notify": {"command": "echo {kind} {id} {contact} >> %s" % out}}))
        jid = self.join("--desktop")
        self.cli("start", jid)
        self.cli("done", jid, "--exit", "3", now=T0 + MIN)
        self.assertEqual(out.read_text().split(), ["died", "early", jid, "painter"])

    def test_notices_reach_the_snapshot(self):
        jid = self.join()
        self.cli("start", jid)
        self.cli("done", jid, "--exit", "1", now=T0 + MIN)
        snap = json.loads(self.cli("show", "--json", now=T0 + 2 * MIN).stdout)
        self.assertEqual([n["id"] for n in snap["notices"]], [jid])
        from gpuq import cli as g
        alerts = g.fleet_alerts([("h", snap, None)], T0 + 2 * MIN)
        self.assertTrue(any("died early" in a[1] and jid in a[1] for a in alerts), alerts)

    def test_beat_metrics_land_in_history(self):
        jid = self.join()
        self.cli("start", jid)
        self.cli("beat", jid, "--mem", "5.86", "--util", "80", now=T0 + 20 * MIN)
        self.cli("beat", jid, "--mem", "8.79", "--util", "100", now=T0 + 40 * MIN)
        self.cli("done", jid, "--exit", "0", now=T0 + 60 * MIN)
        last = self.history()[-1]
        self.assertEqual((last["event"], last["mem_peak_mib"], last["util_avg"]), ("done", 9000, 90))
        self.assertIn("load 90%", last["note"])

    def test_run_saves_the_log_and_reports_the_exit_code(self):
        jid = self.join()
        r = self.cli("run", jid, "--every", "0.1", "--", "sh", "-c", "echo hello-from-job; exit 3")
        self.assertEqual(r.returncode, 3, r.stderr)
        log = self.state / "logs" / ("%s.log" % jid)
        self.assertIn("hello-from-job", log.read_text())
        self.assertIn("hello-from-job", r.stdout)         # still shown on the terminal
        got = self.inbox("--contact", "painter")
        self.assertEqual((got[0]["exit"], got[0]["log"]), (3, str(log)))

    def test_run_without_log(self):
        jid = self.join()
        r = self.cli("run", jid, "--every", "0.1", "--log", "-", "--", "sh", "-c", "echo x")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((self.state / "logs").exists())


class Mcp(unittest.TestCase):
    def test_notices_come_back_on_the_next_call(self):
        from gpuq import mcp
        calls = []

        def fake(args, host=None):
            calls.append(args)
            if args[0] == "join":
                return 0, json.dumps({"ok": True, "id": "ab12"}), ""
            if args[0] == "inbox":
                return 0, json.dumps([{"id": "ab12", "kind": "died early", "job": "lora",
                                       "text": "exit 1 after 1m", "log": "/x.log"}]), ""
            return 0, json.dumps({"ok": True}), ""
        srv = mcp.Server(cli_runner=fake)
        srv.handle({"id": 1, "method": "tools/call", "params": {"name": "gpuq_join",
                    "arguments": {"owner": "painter", "job": "lora"}}})
        r = srv.handle({"id": 2, "method": "tools/call", "params": {"name": "gpuq_show", "arguments": {}}})
        res = r["result"]["structuredContent"]
        self.assertEqual(res["notices"][0]["state"], "died early")
        self.assertIn("/x.log", res["notices"][0]["message"])
        self.assertTrue(any(c[:5] == ["inbox", "--json", "--ack", "--contact", "painter"] and "--hosts" in c
                            for c in calls), calls)

    def test_done_passes_exit_code(self):
        from gpuq import mcp
        calls = []
        srv = mcp.Server(cli_runner=lambda a, host=None: (calls.append(a), (0, '{"ok": true}', ""))[1])
        srv.call("gpuq_done", {"id": "ab12", "exit_code": 2})
        self.assertIn("--exit=2", calls[0])


if __name__ == "__main__":
    unittest.main()
