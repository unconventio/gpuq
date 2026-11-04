"""AMD support: cards come from sysfs, usage from DRM fdinfo. Fixture trees below copy the shape
(and the numbers) of a real Radeon 880M laptop (Strix Point, amdgpu, no amd-smi); the discrete
card is synthetic, modeled on what amdgpu exposes for an RX 7900 XTX."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GPUQ_STATE", tempfile.mkdtemp(prefix="gpuq-test-"))
os.environ.setdefault("GPUQ_CONFIG", os.path.join(os.environ["GPUQ_STATE"], "no-config.json"))
os.environ.setdefault("GPUQ_COMPAT_LOCK", "")
from gpuq import cli as g  # noqa: E402

PCI_IDS = """\
# fixture
1002  Advanced Micro Devices, Inc. [AMD/ATI]
\t150e  Strix [Radeon 880M / 890M]
\t744c  Navi 31 [Radeon RX 7900 XTX]
8086  Intel Corporation
\t150e  82580 Gigabit Network Connection
"""
GIB = 1 << 30


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def add_card(sys_root, n, vendor, dev, slot, vram, gtt, driver="amdgpu", unique_id=None, busy=None, render=128):
    d = sys_root / "class" / "drm" / ("card%d" % n) / "device"
    write(d / "vendor", vendor + "\n")
    write(d / "device", dev + "\n")
    write(d / "uevent", "DRIVER=%s\nPCI_ID=1002:%s\nPCI_SLOT_NAME=%s\n" % (driver, dev[2:].upper(), slot))
    write(d / "mem_info_vram_total", "%d\n" % vram)
    write(d / "mem_info_gtt_total", "%d\n" % gtt)
    (d / "drm" / ("renderD%d" % render)).mkdir(parents=True)
    if unique_id:
        write(d / "unique_id", unique_id + "\n")
    if busy is not None:
        write(d / "gpu_busy_percent", "%d\n" % busy)


def add_kfd(sys_root, renders):
    """node 0 = the CPU; then one GPU node per render minor, in order."""
    nodes = sys_root / "class" / "kfd" / "kfd" / "topology" / "nodes"
    write(nodes / "0" / "properties", "cpu_cores_count 20\nsimd_count 0\ndrm_render_minor 0\n")
    for i, minor in enumerate(renders, 1):
        write(nodes / str(i) / "properties", "cpu_cores_count 0\nsimd_count 24\ndrm_render_minor %d\n" % minor)


def add_proc(proc, pid, comm, client, pdev, vram_kib, gtt_kib, fd="4"):
    write(proc / str(pid) / "comm", comm + "\n")
    write(proc / str(pid) / "fdinfo" / fd,
          "pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t%d\ndrm-pdev:\t%s\n"
          "drm-memory-vram:\t%d KiB\ndrm-memory-gtt: \t%d KiB\n" % (client, pdev, vram_kib, gtt_kib))


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gpuq-amd-"))
        self.sys, self.proc = self.tmp / "sys", self.tmp / "proc"
        self.sys.mkdir(), self.proc.mkdir()
        write(self.tmp / "pci.ids", PCI_IDS)
        self.saved = (g.SYSFS, g.PROC, g.PCI_IDS, os.environ.pop("GPUQ_CARDS", None))
        g.SYSFS, g.PROC, g.PCI_IDS = self.sys, self.proc, (str(self.tmp / "pci.ids"),)
        self._nv = g._nvidia_cards
        g._nvidia_cards = lambda: {}           # these tests are about AMD only

    def tearDown(self):
        g.SYSFS, g.PROC, g.PCI_IDS = self.saved[:3]
        g._nvidia_cards = self._nv

    def laptop(self):
        """Radeon 880M: 512 MiB carve-out, 24 GiB of shared memory (the numbers measured on one)."""
        add_card(self.sys, 1, "0x1002", "0x150e", "0000:65:00.0", 536870912, 25769803776, busy=97)
        add_kfd(self.sys, [128])

    def desktop(self):
        add_card(self.sys, 0, "0x1002", "0x744c", "0000:03:00.0", 24 * GIB, 32 * GIB,
                 unique_id="0x1234ABCD5678EF01")
        add_card(self.sys, 1, "0x1002", "0x150e", "0000:65:00.0", 536870912, 25769803776, render=129)
        add_kfd(self.sys, [128, 129])


class Detect(Fixture):
    def test_integrated_gpu(self):
        self.laptop()
        cards = g.amd_cards()
        self.assertEqual(list(cards), ["880M"])
        c = cards["880M"]
        self.assertTrue(c["integrated"])
        self.assertEqual(c["mib"], 24576)              # the shared memory, not the 512 MiB carve-out
        self.assertEqual((c["env_var"], c["env_value"]), ("ROCR_VISIBLE_DEVICES", "0"))
        self.assertEqual(c["uuid"], "0000:65:00.0")    # no unique id: the PCI address stands in

    def test_discrete_and_integrated(self):
        self.desktop()
        cards = g.amd_cards()
        self.assertEqual(list(cards), ["7900", "880M"])
        self.assertFalse(cards["7900"]["integrated"])
        self.assertEqual(cards["7900"]["mib"], 24576)
        self.assertEqual(cards["7900"]["env_value"], "GPU-1234abcd5678ef01")
        self.assertEqual(cards["880M"]["env_value"], "1")      # second ROCm device
        self.assertEqual(g.default_card(cards), "7900")        # discrete beats shared memory

    def test_ignores_other_vendors_and_drivers(self):
        add_card(self.sys, 0, "0x10de", "0x2204", "0000:01:00.0", GIB, GIB, driver="nvidia")
        add_card(self.sys, 1, "0x1002", "0x150e", "0000:65:00.0", GIB, GIB, driver="radeon")
        self.assertEqual(g.amd_cards(), {})

    def test_no_sysfs_no_cards(self):
        self.assertEqual(g.amd_cards(), {})

    def test_unknown_device_gets_a_fallback_name(self):
        add_card(self.sys, 0, "0x1002", "0xbeef", "0000:03:00.0", 8 * GIB, 8 * GIB)
        self.assertEqual(list(g.amd_cards()), ["amd-beef"])

    def test_no_kfd_means_no_pin(self):
        add_card(self.sys, 1, "0x1002", "0x150e", "0000:65:00.0", 536870912, 25769803776)
        c = g.amd_cards()["880M"]
        self.assertEqual(c["env_value"], "")
        q = {"lines": {"880M": [{"id": "ab12"}]}, "cards": {"880M": c}}
        self.assertIsNone(g.card_env(q, "ab12"))

    def test_gpu_cards_merges_after_nvidia(self):
        self.laptop()
        old = g._nvidia_cards
        g._nvidia_cards = lambda: {"3090": {"uuid": "GPU-x", "mib": 24576}}
        try:
            self.assertEqual(list(g.gpu_cards()), ["3090", "880M"])
        finally:
            g._nvidia_cards = old


class Usage(Fixture):
    def test_fdinfo_jobs_and_busy(self):
        self.laptop()
        add_proc(self.proc, 100, "ollama", 7, "0000:65:00.0", 400000, 6 * 1048576)      # ~6.4 GiB
        add_proc(self.proc, 101, "kwin", 9, "0000:65:00.0", 12, 2048)                   # desktop: ignored
        add_proc(self.proc, 102, "ollama2", 3, "0000:99:00.0", 0, 9 * 1048576)         # other card
        # one process, two fds on the same client: counted once
        add_proc(self.proc, 100, "ollama", 7, "0000:65:00.0", 400000, 6 * 1048576, fd="9")
        use = g.amd_usage(g.amd_cards())["880M"]
        self.assertEqual(len(use["jobs"]), 1)
        self.assertTrue(use["jobs"][0].startswith("ollama 6."), use["jobs"])
        self.assertEqual(use["busy"], 97)

    def test_forked_workers_share_one_client(self):
        """PyTorch data-loader workers inherit the trainer's DRM file: count it once, by the parent."""
        self.laptop()
        add_proc(self.proc, 200, "trainer", 42, "0000:65:00.0", 3448, 5 * 1048576)
        for pid in range(201, 219):
            add_proc(self.proc, pid, "pt_data_worker", 42, "0000:65:00.0", 3448, 5 * 1048576)
        jobs = g.amd_usage(g.amd_cards())["880M"]["jobs"]
        self.assertEqual(jobs, ["trainer 5.0G"], jobs)

    def test_card_usage_and_residents(self):
        self.laptop()
        add_proc(self.proc, 100, "whisper-server", 7, "0000:65:00.0", 0, 2 * 1048576)
        add_proc(self.proc, 101, "worker", 8, "0000:65:00.0", 0, 3 * 1048576)
        use = g.card_usage(g.amd_cards(), {"880M": ["whisper-server"]})
        self.assertEqual(use["880M"]["residents"][0][:14], "whisper-server")
        self.assertEqual([j.split()[0] for j in use["880M"]["jobs"]], ["worker"])

    def test_snapshot_shows_usage_without_nvidia(self):
        self.laptop()
        add_proc(self.proc, 100, "worker", 7, "0000:65:00.0", 0, 3 * 1048576)
        with tempfile.TemporaryDirectory() as st:
            old = g.STATE
            g.STATE = Path(st)
            try:
                snap = g.local_snapshot(1_760_000_000.0)
            finally:
                g.STATE = old
        self.assertEqual(snap["loaded"] and snap["loaded"][0].split()[0], "worker")
        view = g.per_card(snap)[0]
        self.assertIn("shared", g.header(dict(snap, card="880M"), snap["now"]))
        self.assertIn("processes on this card: worker", "\n".join(g.warnings(dict(snap, busy=97))))
        self.assertIn("busy 97%", "\n".join(g.warnings(dict(snap, busy=97))))
        self.assertTrue(view)


class Migrate(Fixture):
    def test_placeholder_line_moves_to_the_detected_card(self):
        self.laptop()
        state = self.tmp / "state"
        write(state / "queue.json", json.dumps({"lines": {"gpu": [
            {"id": "ab12", "owner": "x", "job": "j", "why": "", "models": [], "est_min": 60, "max_min": None,
             "progress": "", "state": "running", "joined": 1.0, "started": 2.0, "heartbeat": 3.0,
             "head_since": None, "skips": 0, "important": False}]}}))
        old = g.STATE
        g.STATE = state
        try:
            with g.store() as Q:
                self.assertEqual(list(Q["lines"]), ["880M"])
                self.assertEqual(Q["lines"]["880M"][0]["state"], "running")
            data = json.loads((state / "queue.json").read_text())
            self.assertEqual(list(data["lines"]), ["880M"])
        finally:
            g.STATE = old


class EndToEnd(Fixture):
    def cli(self, *args, **kw):
        env = dict(os.environ, GPUQ_STATE=str(self.tmp / "state"), GPUQ_SYSFS=str(self.sys),
                   GPUQ_PROC=str(self.proc), GPUQ_CONFIG=str(self.tmp / "none.json"), PATH="/nonexistent",
                   GPUQ_COMPAT_LOCK="", PYTHONPATH=str(ROOT))
        env.pop("GPUQ_CARDS", None)
        return subprocess.run([sys.executable, "-m", "gpuq", *args], capture_output=True, text=True,
                              env=env, **kw)

    def test_two_amd_cards_pin_by_rocr(self):
        self.desktop()
        jid = self.cli("join", "--owner", "a", "--job", "j", "--gpu", "880M").stdout.strip()
        r = self.cli("start", jid)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "ROCR_VISIBLE_DEVICES=1"), r.stderr)
        self.cli("done", jid)
        jid = self.cli("join", "--owner", "a", "--job", "j", "--gpu", "7900").stdout.strip()
        self.assertEqual(self.cli("env", jid).stdout.strip(), "ROCR_VISIBLE_DEVICES=GPU-1234abcd5678ef01")

    def test_run_sets_rocr_not_cuda(self):
        self.desktop()
        jid = self.cli("join", "--owner", "a", "--job", "j", "--gpu", "880M").stdout.strip()
        r = self.cli("run", jid, "--", "/bin/sh", "-c", "echo R=$ROCR_VISIBLE_DEVICES C=${CUDA_VISIBLE_DEVICES-unset}")
        self.assertIn("R=1 C=unset", r.stdout, r.stderr)

    def test_single_amd_card_queue_works(self):
        self.laptop()
        jid = self.cli("join", "--owner", "a", "--job", "j", "--est", "5").stdout.strip()
        self.assertEqual(self.cli("start", jid).returncode, 0)
        out = self.cli("show", "--hosts", g.HOST).stdout
        self.assertIn("RUNNING", out)
        self.assertEqual(json.loads(self.cli("show", "--json").stdout)["cards"]["880M"]["integrated"], True)


if __name__ == "__main__":
    unittest.main()
