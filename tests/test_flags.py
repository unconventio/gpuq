"""The flag scheme: --host anywhere, no abbreviations, lists that repeat, aliases, edit by id only,
--json on the resident commands, and the MCP tools that mirror the CLI."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
T0 = 1_760_000_000.0


class Cli(unittest.TestCase):
    cards = {"gpu": {"uuid": "GPU-one", "mib": 8192}}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.state = self.dir / "state"
        self.cfg = self.dir / "config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *args):
        env = dict(os.environ, GPUQ_STATE=str(self.state), GPUQ_COMPAT_LOCK="", GPUQ_NOW=str(T0),
                   GPUQ_CARDS=json.dumps(self.cards), GPUQ_CONFIG=str(self.cfg), GPUQ_DESKTOP="0")
        return subprocess.run([sys.executable, "-m", "gpuq", *args], capture_output=True, text=True,
                              env=env, cwd=str(ROOT), timeout=60)

    def join(self, *extra):
        r = self.cli("join", "--owner", "painter", "--job", "lora", *extra)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def entry(self, jid):
        q = json.loads((self.state / "queue.json").read_text())
        return [x for line in q["lines"].values() for x in line if x["id"] == jid][0]

    def test_host_after_the_command_is_the_same_flag(self):
        jid = self.join()
        here = socket.gethostname()      # this host: runs locally, no ssh
        for args in (("env", jid, "--host", here), ("--host", here, "env", jid), ("pause", "--host", here)):
            r = self.cli(*args)
            self.assertEqual(r.returncode, 0, (args, r.stderr))

    def test_abbreviations_are_refused(self):
        r = self.cli("show", "--hos", "nowhere")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unrecognized", r.stderr)
        r = self.cli("join", "--owner", "a", "--job", "b", "--pri", "chill")
        self.assertEqual(r.returncode, 2)

    def test_lists_take_commas_and_repeats(self):
        jid = self.join("--models", "a", "--models", "b,c", "--contact", "one", "--contact", "two")
        r = self.cli("edit", jid, "--tag", "x,y", "--tag", "z")      # edit doesn't check the card's tags
        self.assertEqual(r.returncode, 0, r.stderr)
        e = self.entry(jid)
        self.assertEqual((e["models"], e["tags"], e["contact"]), (["a", "b", "c"], ["x", "y", "z"], ["one", "two"]))

    def test_aliases(self):
        jid = self.join("--limit", "90", "--card", "gpu", "--notify")
        e = self.entry(jid)
        self.assertEqual((e["max_min"], e["gpu"], e["desktop"]), (90, "gpu", True))
        out = self.dir / "c.json"
        env_cfg = self.cfg
        self.cfg = out
        r = self.cli("init", "--dry-run")
        self.cfg = env_cfg
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(out.exists())

    def test_edit_takes_ids_only(self):
        self.join()
        r = self.cli("edit", "1", "--job", "x")
        self.assertEqual(r.returncode, 4)
        self.assertIn("no entry", r.stderr)

    def test_show_json_refuses_other_hosts(self):
        r = self.cli("show", "--json", "--hosts", "elsewhere")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--host", r.stderr)
        self.assertEqual(self.cli("show", "--json").returncode, 0)

    def test_local_means_this_host(self):
        r = self.cli("inbox", "--json", "--local", "--hosts", "elsewhere")
        self.assertEqual((r.returncode, json.loads(r.stdout)), (0, []), r.stderr)

    def test_free_json_is_an_envelope(self):
        r = self.cli("free", "--json", "--dry-run", "--card", "gpu")
        res = json.loads(r.stdout)
        self.assertEqual((res["state"], res["card"], res["dry_run"]), ("ok", "gpu", True))
        self.assertTrue(res["lines"])

    def test_help_lists_exit_codes(self):
        r = self.cli("--help")
        self.assertIn("exit codes:", r.stdout)
        cmds = r.stdout[r.stdout.index("commands:"):]
        self.assertLess(cmds.index("  join "), cmds.index("  show "))


class Desktop(unittest.TestCase):
    def test_desktop_apps_are_not_unqueued_use(self):
        from gpuq import cli
        self.assertTrue(cli._desktop_app("obs"))
        self.assertFalse(cli._desktop_app("python"))
        snap = {"entries": [], "loaded": [], "desktop": ["obs 1.1G"]}
        w = cli.warnings(snap)
        self.assertIn("desktop: obs 1.1G", w[0])
        self.assertFalse(any("unqueued" in x for x in w))


class Forward(unittest.TestCase):
    def test_host_is_dropped_only_before_the_command_separator(self):
        from gpuq import cli
        with mock.patch.object(cli.subprocess, "call", return_value=0) as call:
            cli._forward(["run", "ab12", "--host", "otherbox", "--", "tool", "--host", "x"], "otherbox")
        cmd = call.call_args[0][0][-1]
        self.assertTrue(cmd.endswith("run ab12 -- tool --host x"), cmd)


class Mcp(unittest.TestCase):
    def server(self):
        from gpuq import mcp
        self.calls = []
        return mcp.Server(cli_runner=lambda a, host=None: (self.calls.append(a), (0, '{"ok": true}', ""))[1])

    def test_edit_maps_fields_and_clears(self):
        srv = self.server()
        srv.call("gpuq_edit", {"id": "ab12", "contact": "artist", "desktop": False, "log": "",
                               "priority": "important", "max_min": 600})
        c = self.calls[0]
        self.assertEqual(c[:3], ["edit", "ab12", "--json"])
        for pair in (["--contact", "artist"], ["--log", ""], ["--priority", "important"], ["--max", "600"]):
            i = c.index(pair[0])
            self.assertEqual(c[i:i + 2], pair)
        self.assertIn("--no-desktop", c)
        self.assertIn(("artist", None), srv.watch)

    def test_rm_yield_env_want_an_id(self):
        srv = self.server()
        self.assertFalse(srv.call("gpuq_rm", {"id": "2"})["ok"])
        self.assertEqual(self.calls, [])
        srv.call("gpuq_yield", {"id": "ab12", "to": 3})
        srv.call("gpuq_env", {"id": "ab12"})
        self.assertEqual(self.calls, [["yield", "ab12", "--json", "3"], ["env", "ab12", "--json"]])

    def test_join_priority_name_no_history_and_done_no_notice(self):
        srv = self.server()
        srv.call("gpuq_join", {"owner": "o", "job": "j", "priority": "chill", "no_history": True})
        srv.call("gpuq_done", {"id": "ab12", "no_notice": True})
        self.assertIn("--no-history", self.calls[0])
        self.assertEqual(self.calls[0][self.calls[0].index("--priority") + 1], "chill")
        self.assertIn("--no-notice", self.calls[1])

    def test_every_tool_is_handled(self):
        from gpuq import mcp
        srv = self.server()
        for t in mcp.TOOLS:
            srv.call(t["name"], {"id": "ab12", "owner": "o", "job": "j", "contact": "o"})


if __name__ == "__main__":
    unittest.main()
