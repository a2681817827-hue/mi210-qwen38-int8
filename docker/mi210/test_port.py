# SPDX-License-Identifier: Apache-2.0
"""GPU-free regression checks for the MI210 integration and its routing."""

import ast
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from apply_aiter import patch_checkout
from launch import command

ROOT = Path(__file__).resolve().parents[2]


class PortTests(unittest.TestCase):
    def test_profiles_keep_ptqr_contract_and_tp(self):
        for profile, tp in (("single", 1), ("replicas2", 1), ("tp2", 2)):
            env, argv = command(profile)
            self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], str(tp))
            self.assertEqual(argv[argv.index("--kv-cache-dtype") + 1], "int8_block_g128")
            self.assertEqual(argv[argv.index("--mamba-ssm-cache-dtype") + 1], "float32")
            spec = json.loads(argv[argv.index("--speculative-config") + 1])
            self.assertEqual(spec["kv_cache_dtype"], "int8_block_g128")
            self.assertEqual(spec["num_speculative_tokens"], 13)
            self.assertIn("PTQR", spec["model"])
            self.assertEqual(env["VLLM_ROCM_USE_AITER_CUSTOM_AR"], "0")
            self.assertEqual(env["VLLM_G128_GLUON"], "0")

    def test_draft_path_is_json_escaped(self):
        path = '/models/a "quoted" draft'
        _, argv = command("single", draft=path, spec_tokens=12)
        self.assertEqual(json.loads(argv[-1])["model"], path)
        with self.assertRaises(ValueError):
            command("single", spec_tokens=0)

    def test_source_drift_never_half_applies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "aiter/jit/utils/build_targets.py"
            gen = root / "csrc/ck_gemm_a8w8/gen_instances.py"
            target.parent.mkdir(parents=True)
            gen.parent.mkdir(parents=True)
            original = 'GFX_CU_NUM_MAP = {\n    "gfx908": 120,\n}\n'
            target.write_text(original)
            gen.write_text("# a changed upstream generator\n")
            with self.assertRaises(ValueError):
                patch_checkout(root)
            self.assertEqual(target.read_text(), original)

    def test_patch_is_idempotent_and_check_requires_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "aiter/jit/utils/build_targets.py"
            gen = root / "csrc/ck_gemm_a8w8/gen_instances.py"
            target.parent.mkdir(parents=True)
            gen.parent.mkdir(parents=True)
            target.write_text('GFX_CU_NUM_MAP = {\n    "gfx908": 120,\n}\n')
            gen.write_text('if False: pass\nelif all(gfx == "gfx908" for gfx, _ in get_build_targets()):\n    pass\n# gen_built_combos_header\n')
            with self.assertRaises(ValueError):
                patch_checkout(root, check=True)
            patch_checkout(root)
            self.assertEqual(patch_checkout(root)["changed"], [])
            self.assertEqual(patch_checkout(root, check=True)["changed"], [])

    def test_grouped_kv_constructor_does_not_import_aiter_kernel(self):
        path = ROOT / "vllm/v1/attention/backends/rocm_aiter_unified_attn.py"
        tree = ast.parse(path.read_text())
        impl = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == "RocmAiterUnifiedAttentionImpl")
        constructor = next(n for n in impl.body if isinstance(n, ast.FunctionDef)
                           and n.name == "__init__")
        module = ast.Module(body=[
            ast.parse("from __future__ import annotations").body[0],
            ast.ClassDef(name="Impl", bases=[ast.Name(id="Base", ctx=ast.Load())],
                         keywords=[], body=[constructor], decorator_list=[]),
        ], type_ignores=[])

        class Base:
            def __init__(self, *args, **kwargs):
                pass

        ns = {"Base": Base, "AttentionType": SimpleNamespace(DECODER=1),
              "logger": SimpleNamespace(info_once=lambda *a: None),
              "get_kv_quant_mode": lambda dtype: SimpleNamespace(
                  is_per_token_head=False,
                  is_int8_block=True, int8_block_group=128)}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
        instance = ns["Impl"](6, 256, 0.0625, 1, None, None, "int8_block_g128")
        self.assertIsNone(instance.unified_attention)
        self.assertEqual(instance._block_g, 128)

    def test_grouped_kv_dtype_is_advertised(self):
        tree = ast.parse((ROOT / "vllm/v1/attention/backends/rocm_aiter_unified_attn.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "RocmAiterUnifiedAttentionBackend")
        node = next(n for n in cls.body if isinstance(n, ast.AnnAssign)
                    and n.target.id == "supported_kv_cache_dtypes")
        values = ast.literal_eval(node.value)
        self.assertTrue(all(f"int8_block_g{g}" in values for g in (4, 8, 16, 32, 64, 128)))

    def test_dual_replica_device_isolation_and_router(self):
        env = {**os.environ, "GPU_NODES": "/dev/dri/renderD128 /dev/dri/renderD131"}
        output = subprocess.check_output(
            ["bash", str(ROOT / "scripts/serve_mi210.sh"), "replicas2", "--dry-run"],
            env=env, text=True)
        commands = [shlex.split(line) for line in output.splitlines()]
        self.assertEqual(len(commands), 3)
        for index, node in enumerate(("/dev/dri/renderD128", "/dev/dri/renderD131")):
            devices = [commands[index][i + 1] for i, word in enumerate(commands[index])
                       if word == "--device"]
            self.assertEqual(devices, ["/dev/kfd", node])
        self.assertIn("/opt/int8-vllm/docker/mi210/router.py", commands[2])

    def test_tp2_uses_both_nodes_in_one_engine(self):
        env = {**os.environ, "GPU_NODES": "/dev/dri/renderD128 /dev/dri/renderD131"}
        output = subprocess.check_output(
            ["bash", str(ROOT / "scripts/serve_mi210.sh"), "tp2", "--dry-run"],
            env=env, text=True)
        words = shlex.split(output)
        self.assertEqual(words.count("--device"), 3)
        self.assertEqual(output.count("docker run"), 1)


class RouterTests(unittest.TestCase):
    def test_sse_and_auth_survive_proxy_and_replicas_alternate(self):
        import httpx
        from fastapi.testclient import TestClient
        from router import create_app

        seen = []
        payload = b'data: {"choices":[{"text":"hello"}]}\n\ndata: [DONE]\n\n'

        async def handler(request):
            seen.append((request.url.host, request.headers.get("authorization")))
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=httpx.ByteStream(payload))

        app = create_app(["http://worker0", "http://worker1"])
        with TestClient(app) as client:
            app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            for _ in range(2):
                response = client.post("/v1/completions", json={"stream": True},
                                       headers={"authorization": "Bearer test"})
                self.assertEqual(response.content, payload)
                self.assertEqual(response.headers["content-type"], "text/event-stream")
        self.assertEqual(seen, [("worker0", "Bearer test"), ("worker1", "Bearer test")])

    def test_unavailable_worker_returns_502_and_releases_slot(self):
        import httpx
        from fastapi.testclient import TestClient
        from router import create_app

        async def handler(request):
            raise httpx.ConnectError("offline", request=request)

        app = create_app(["http://worker0", "http://worker1"])
        with TestClient(app) as client:
            app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            self.assertEqual(client.post("/v1/completions").status_code, 502)
            self.assertEqual(client.get("/health").status_code, 503)


if __name__ == "__main__":
    unittest.main()
