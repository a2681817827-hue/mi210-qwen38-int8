"""Exercise MI210 device isolation on a simulated mixed-GPU host."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class DeviceIsolationTests(unittest.TestCase):
    def launch(self, env, script="serve_mi210_long_context.sh", profile="graph-c8"):
        return subprocess.run(
            ["bash", str(ROOT / "scripts" / script), profile, "--dry-run"],
            env=env, text=True, capture_output=True)

    def fixture(self, root):
        dri, sysfs = root / "dri", root / "sysfs"
        dri.mkdir()
        for index, vendor, device in ((128, "0x1002", "0x7550"),
                                      (129, "0x1002", "0x740f"),
                                      (130, "0x1002", "0x740c"),
                                      (131, "0x1234", "0x740f")):
            name = f"renderD{index}"
            (dri / name).touch()
            path = sysfs / name / "device"
            path.mkdir(parents=True)
            (path / "vendor").write_text(vendor)
            (path / "device").write_text(device)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("MI210_", "GPU_NODES"))}
        env.update(MI210_DRI_ROOT=str(dri), MI210_SYSFS_ROOT=str(sysfs),
                   MI210_API_KEY="test-only")
        return dri, env

    def test_single_selects_only_cdna2_and_reindexes_visibility(self):
        with tempfile.TemporaryDirectory() as directory:
            dri, env = self.fixture(Path(directory))
            for index, name in ((0, "renderD129"), (1, "renderD130")):
                result = self.launch({**env, "MI210_GPU": str(index)})
                self.assertEqual(result.returncode, 0, result.stderr)
                words = shlex.split(result.stdout)
                devices = [word for word in words if word.startswith("--device=")]
                self.assertEqual(devices, ["--device=/dev/kfd", f"--device={dri / name}"])
                self.assertIn("HIP_VISIBLE_DEVICES=0", words)
            self.assertEqual(self.launch({**env, "MI210_GPU": "2"}).returncode, 2)

    def test_multi_card_launcher_uses_same_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            dri, env = self.fixture(Path(directory))
            result = self.launch(env, "serve_mi210.sh", "tp2")
            self.assertEqual(result.returncode, 0, result.stderr)
            words = shlex.split(result.stdout)
            devices = [words[i + 1] for i, word in enumerate(words) if word == "--device"]
            self.assertEqual(devices, ["/dev/kfd", str(dri / "renderD129"),
                                      str(dri / "renderD130")])

    def test_no_gpu_and_whole_directory_override_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            dri, env = self.fixture(Path(directory))
            for node in dri.iterdir():
                node.unlink()
            for settings in (env, {**env, "GPU_NODES": "/dev/dri"}):
                result = self.launch(settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("docker run", result.stdout)


if __name__ == "__main__":
    unittest.main()
