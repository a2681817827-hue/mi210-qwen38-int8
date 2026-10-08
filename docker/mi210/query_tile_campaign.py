"""Gate an isolated source overlay; never modify a running worker's imports."""

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    args.output_dir.mkdir(parents=True)
    import torch
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx90a":
        raise RuntimeError("MI210 required")
    package = importlib.import_module("vllm.v1.attention.ops")
    for name in ("mi210_split_tuning", "gfx90a_g128_gluon_decode",
                 "triton_unified_attention"):
        qualified = "vllm.v1.attention.ops." + name
        spec = importlib.util.spec_from_file_location(qualified, args.source_dir / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        setattr(package, name, module)
        spec.loader.exec_module(module)
    from micro import check_attention
    native = sys.modules["vllm.v1.attention.ops.gfx90a_g128_gluon_decode"]
    original = native.g128_core_cdna2

    class Tracker:
        launches = 0

        def __getitem__(self, grid):
            self.launches += 1
            return original[grid]

    tracker = Tracker()
    native.g128_core_cdna2 = tracker
    os.environ.update({"VLLM_UA_TILE": "32", "VLLM_UA_3D_MAXQ": "16",
                       "VLLM_G128_GLUON": "64", "VLLM_G128_GLUON_MMA": "bf16",
                       "VLLM_G128_DECODE_CDNA2": "1",
                       "VLLM_G128_DECODE_CDNA2_TILE": "64",
                       "VLLM_MI210_FLASH_SPLITS": "64"})
    for depth in (33, 4097, 48081, 65537):
        for query in (1, 10, 14, 16):
            for tile in (10, 16):
                os.environ["VLLM_MI210_G128_QUERY_TILE"] = str(tile)
                before = tracker.launches
                row = {"context": depth, "query": query, "query_tile": tile,
                       "splits": 64, "block_size": 1728, "heads": 24, "kv_heads": 4}
                result = check_attention(256, 24, 4, query, True, True,
                                         depth, 1728, benchmark_repeats=9,
                                         segment_capacity=64, benchmark_mode="graph")
                if tracker.launches <= before:
                    raise RuntimeError("Native candidate not exercised")
                row.update(result, numeric="PASS", candidate_launches=tracker.launches - before)
                with (args.output_dir / "operators.jsonl").open("a") as file:
                    file.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                torch.cuda.empty_cache()
    native.g128_core_cdna2 = original
    for tile in (10, 16):
        for splits in (0, 32, 64):
            sys.argv = ["graph_split_gate.py", "--splits", str(splits),
                        "--query-tile", str(tile), "--max-query", "14",
                        "--output", str(args.output_dir / f"graph-q{tile}-s{splits}.jsonl")]
            runpy.run_path(str(Path(__file__).with_name("graph_split_gate.py")), run_name="__main__")
            torch.cuda.empty_cache()
    (args.output_dir / "status.json").write_text(json.dumps({
        "operator_numerics": "PASS", "graph_replay": "PASS",
        "operator_cases": 32, "ragged_graph_cases": 6,
        "serving_validated": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
