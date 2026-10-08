"""Gate CK decode kernels at actual M shapes, FP32 scales and BF16 output."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

IDS = (31, 33, 34, 49, 50, 59, 61, 63, 65, 68, 79, 80, 82, 84)
SHAPES = ((16384, 5120), (34816, 5120), (5120, 17408),
          (5120, 6144), (5120, 8192), (14336, 5120), (248320, 5120))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 10, 14],
                        help="Exact decode/verification M; defaults cover NS7/9/13")
    parser.add_argument("--timing", choices=("eager", "graph"), default="graph")
    parser.add_argument("--aiter-root", type=Path, default=Path("/opt/aiter"))
    args = parser.parse_args()
    if any(m <= 0 for m in args.rows) or len(set(args.rows)) != len(args.rows):
        parser.error("--rows must contain unique positive integers")
    if args.output_dir.exists():
        parser.error("Use a fresh directory")
    args.output_dir.mkdir(parents=True)
    os.environ["AITER_CK_INSTANCE_LIST"] = ",".join(map(str, IDS))
    os.environ.setdefault("MAX_JOBS", "4")
    # Build only the exact serving template; other dtypes are not measured here.
    os.environ.setdefault("AITER_CK_DTYPES", "I8xF32xB16")
    import torch
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_ck, gemm_a8w8_tune
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Isolate one idle MI210; do not tune alongside serving")
    properties = torch.cuda.get_device_properties(0)
    if properties.gcnArchName.split(":")[0] != "gfx90a":
        raise RuntimeError("MI210 required")
    # Record the exact kernel-ID/name mapping used by this checkout.
    path = args.aiter_root / "csrc/ck_gemm_a8w8/gemm_a8w8_common.py"
    spec = importlib.util.spec_from_file_location("ck_common", path)
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    names = {i: common.kernels_list[i].name for i in IDS}
    source_paths = [args.aiter_root / relative for relative in (
        "csrc/ck_gemm_a8w8/gemm_a8w8_common.py",
        "csrc/ck_gemm_a8w8/gen_instances.py",
        "csrc/ck_gemm_a8w8/gemm_a8w8_tune.cu",
        "aiter/ops/gemm_op_a8w8.py")]
    source_hashes = {str(p.relative_to(args.aiter_root)):
                     hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    torch.manual_seed(37)
    selections = []

    def check(y, ref):
        torch.testing.assert_close(y, ref, rtol=0.008, atol=0.004)
        relative = (torch.linalg.vector_norm(y.float() - ref.float()) /
                    torch.linalg.vector_norm(ref.float())).item()
        assert relative < 0.003, relative
        return relative

    def timing(run, y, ref):
        for _ in range(3):
            run()
        check(y, ref)
        if args.timing == "graph":
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            # Overwrite output to ensure replay actually computes it.
            y.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check(y, ref)
            run = graph.replay
        times = []
        for _ in range(5):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(5):
                run()
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end) / 5)
        check(y, ref)
        return statistics.median(times), times

    for m, n, k in ((m, n, k) for m in args.rows for n, k in SHAPES):
        x = torch.randint(-64, 65, (m, k), dtype=torch.int8, device="cuda")
        w = torch.randint(-64, 65, (n, k), dtype=torch.int8, device="cuda")
        xs = torch.rand(m, 1, dtype=torch.float32, device="cuda") * 0.01
        ws = torch.rand(n, 1, dtype=torch.float32, device="cuda") * 0.01
        y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        ref = torch.empty_like(y)
        for start in range(0, n, 4096):
            ref[:, start:start + 4096] = ((x.float() @ w[start:start + 4096].float().T) *
                                         xs * ws[start:start + 4096].T).to(y.dtype)
        baseline, baseline_raw = timing(
            lambda: gemm_a8w8_ck(x, w, xs, ws, y, None, 0), y, ref)
        best = {"M": m, "N": n, "K": k, "baseline_ms": baseline,
                "baseline_raw_ms": baseline_raw, "timing": args.timing,
                "baseline_numeric": "PASS",
                "graph_replay": "PASS" if args.timing == "graph" else "NOT_RUN",
                "selected_ms": baseline, "kernel_id": None, "splitK": 0}
        for kernel in IDS:
            for split in (0, 1, 2, 3):
                row = {"M": m, "N": n, "K": k, "kernel_id": kernel,
                       "kernel_name": names[kernel], "splitK": split,
                       "scale_dtype": "float32", "output_dtype": "bfloat16",
                       "timing": args.timing}
                try:
                    def run():
                        gemm_a8w8_tune(x, w, xs, ws, y, kernel, split)
                    run()
                    relative = check(y, ref)
                    ms, raw = timing(run, y, ref)
                    row.update(numeric="PASS", median_ms=ms, raw_ms=raw,
                               relative_l2=relative,
                               graph_replay="PASS" if args.timing == "graph" else "NOT_RUN")
                    if ms < best["selected_ms"] * 0.98:
                        best.update(selected_ms=ms, kernel_id=kernel,
                                    kernel_name=names[kernel], splitK=split, raw_ms=raw)
                except RuntimeError as error:
                    if "not supported" not in str(error):
                        raise
                    row.update(numeric="UNSUPPORTED", error=str(error)[:200])
                except AssertionError as error:
                    row.update(numeric="FAIL", error=str(error)[:200])
                with (args.output_dir / "trials.jsonl").open("a") as file:
                    file.write(json.dumps(row) + "\n")
                if row["numeric"] == "FAIL":
                    raise RuntimeError(f"CK numeric/replay gate failed: {row}")
        best["speedup"] = baseline / best["selected_ms"]
        if best["speedup"] < 1.05:
            best.update(kernel_id=None, selected_ms=baseline, speedup=1.0, splitK=0)
            best.pop("kernel_name", None)
            best.pop("raw_ms", None)
        selections.append(best)
        (args.output_dir / "selections.json").write_text(json.dumps(
            {"arch": "gfx90a", "cu_num": properties.multi_processor_count,
             "timing": args.timing, "rows": args.rows, "scale_dtype": "float32",
             "aiter_source_sha256": source_hashes, "kernel_names": names,
             "output_dtype": "bfloat16", "serving_validated": False,
             "status": "PARTIAL" if len(selections) < len(args.rows) * len(SHAPES)
                       else "OPERATOR_GATES_COMPLETE",
             "dispatch_note": "selections.json is a tuning report; the current CK "
                              "serving wrapper does not consume its kernel_id. "
                              "Do not claim a serving speedup from this file.",
             "selections": selections}, indent=2) + "\n")
        print(json.dumps(best), flush=True)
        del x, w, xs, ws, y, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
