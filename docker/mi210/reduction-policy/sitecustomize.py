"""Opt-in reproducible Inductor reduction choices from measured MI210 control."""

import os

if os.environ.get("VLLM_MI210_REDUCTION_POLICY", "0") == "1":
    import ast
    import hashlib
    import json
    from pathlib import Path

    import triton
    from torch._inductor.runtime import triton_heuristics

    manifest = Path("/opt/int8-vllm/docker/mi210/tuning/inductor-reduction-baseline-20261008.json")
    payload = manifest.read_bytes()
    expected = os.environ.get("VLLM_MI210_REDUCTION_POLICY_SHA256")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise SystemExit("MI210 reduction policy fingerprint mismatch")
    table = json.loads(payload)["table"]
    original = triton_heuristics.lookup_autotune_config
    reported = set()

    def lookup(size_hints, fn):
        node = ast.parse(fn.src).body[0]
        node.decorator_list = []
        node.name = "kernel"
        digest = hashlib.sha256(ast.dump(node).encode()).hexdigest()
        key = json.dumps([fn.__name__, size_hints, digest],
                         sort_keys=True, separators=(",", ":"))
        config = table.get(key)
        if config is None:
            return original(size_hints, fn)
        if key not in reported:
            print(f"[MI210-REDUCTION-POLICY] {fn.__name__} "
                  f"hints={size_hints} config={config}", flush=True)
            reported.add(key)
        return triton.Config(
            {k: v for k, v in config.items() if "BLOCK" in k},
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )

    triton_heuristics.lookup_autotune_config = lookup
