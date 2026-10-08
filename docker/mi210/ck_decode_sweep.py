"""Numerically check CK decode split-K using the actual FP32/BF16 template."""
import argparse
import json
from pathlib import Path
import statistics

import torch
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_ck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx90a":
        raise RuntimeError("MI210 required")
    torch.manual_seed(29)
    for n, k in ((16384, 5120), (34816, 5120), (5120, 17408),
                 (5120, 6144), (5120, 8192), (14336, 5120), (248320, 5120)):
        x = torch.randint(-64, 65, (args.rows, k), device="cuda", dtype=torch.int8)
        w = torch.randint(-64, 65, (n, k), device="cuda", dtype=torch.int8)
        xs = torch.rand((args.rows, 1), device="cuda", dtype=torch.float32) * 0.01
        ws = torch.rand((n, 1), device="cuda", dtype=torch.float32) * 0.01
        out = torch.empty((args.rows, n), device="cuda", dtype=torch.bfloat16)
        # Bound reference scratch memory for the full 248K vocabulary head.
        ref = torch.empty_like(out)
        for start in range(0, n, 4096):
            chunk = w[start:start + 4096].float()
            ref[:, start:start + 4096] = ((x.float() @ chunk.T) * xs *
                                        ws[start:start + 4096].T).to(out.dtype)
        for split in (0, 1, 2, 3, 4):
            row = {"M": args.rows, "N": n, "K": k, "splitK": split,
                   "scale_dtype": "float32", "output_dtype": "bfloat16"}
            try:
                def run():
                    gemm_a8w8_ck(x, w, xs, ws, out, None, split)
                run()
                # A differently ordered FP32 scale epilogue can cross a BF16
                # rounding midpoint by one ULP. Also enforce a global error
                # bound so a loose absolute tolerance cannot hide bad output.
                torch.testing.assert_close(out, ref, rtol=0.008, atol=0.004)
                relative = (torch.linalg.vector_norm(out.float() - ref.float()) /
                            torch.linalg.vector_norm(ref.float())).item()
                assert relative < 0.003, relative
                for _ in range(3):
                    run()
                times = []
                for _ in range(7):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(5):
                        run()
                    end.record()
                    end.synchronize()
                    times.append(begin.elapsed_time(end) / 5)
                row.update(numeric="PASS", median_ms=statistics.median(times), raw_ms=times)
            except RuntimeError as error:
                row.update(numeric="UNSUPPORTED", error=str(error)[:300])
            except AssertionError as error:
                row.update(numeric="FAIL", error=str(error)[:300])
            with args.output.open("a") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if row["numeric"] == "FAIL":
                raise SystemExit(1)
        del x, w, xs, ws, out, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
