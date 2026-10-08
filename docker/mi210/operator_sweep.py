"""Numerically gate and time grouped INT8 attention on real gfx90a layouts."""
import argparse
import json
import os
from pathlib import Path

from micro import check_attention

CONTROLS = [
    "VLLM_G128_GLUON", "VLLM_G128_PREFILL_GLUON", "VLLM_G128_DRAFT_GLUON",
    "VLLM_G128_PREFILL_PACKED", "VLLM_G128_PREFILL_GLUON_MMA",
    "VLLM_UA_PREFILL_BLOCKM", "VLLM_UA_PREFILL_TILE",
    "VLLM_UA_PREFILL_WARPS", "VLLM_UA_PREFILL_STAGES", "VLLM_UA_TILE",
    "VLLM_G128_PREFILL_CDNA2",
    "VLLM_G128_PREFILL_CDNA2_TILE", "VLLM_G128_DECODE_CDNA2",
    "VLLM_G128_GLUON_MMA", "VLLM_G128_DECODE_CDNA2_TILE",
    "VLLM_MI210_TP2_CDNA2", "VLLM_MI210_G128_QUERY_TILE",
]
ARMS = {
    "generic_decode_tile32": {"VLLM_UA_TILE": "32"},
    "generic_decode_tile64": {"VLLM_UA_TILE": "64"},
    "gluon_cdna2_decode_tile64": {"VLLM_UA_TILE": "32", "VLLM_G128_GLUON": "64",
                          "VLLM_G128_GLUON_MMA": "bf16",
                          "VLLM_G128_DECODE_CDNA2": "1",
                          "VLLM_G128_DECODE_CDNA2_TILE": "64"},
    "generic": {},
    "tile64_m64": {"VLLM_UA_PREFILL_BLOCKM": "64", "VLLM_UA_PREFILL_TILE": "64",
                   "VLLM_UA_PREFILL_WARPS": "4", "VLLM_UA_PREFILL_STAGES": "1"},
    "tile64_m128": {"VLLM_UA_PREFILL_BLOCKM": "128", "VLLM_UA_PREFILL_TILE": "64",
                    "VLLM_UA_PREFILL_WARPS": "4", "VLLM_UA_PREFILL_STAGES": "1"},
    "gluon_bf16": {"VLLM_G128_PREFILL_GLUON": "1",
                   "VLLM_G128_PREFILL_GLUON_MMA": "bf16"},
    "gluon_packed_bf16": {"VLLM_G128_PREFILL_GLUON": "1",
                          "VLLM_G128_PREFILL_GLUON_MMA": "bf16",
                          "VLLM_G128_PREFILL_PACKED": "1"},
    "gluon_cdna2_bf16": {"VLLM_G128_PREFILL_GLUON": "1",
                         "VLLM_G128_PREFILL_GLUON_MMA": "bf16",
                         "VLLM_G128_PREFILL_CDNA2": "1"},
    "gluon_cdna2_tile64": {"VLLM_G128_PREFILL_GLUON": "1",
                          "VLLM_G128_PREFILL_GLUON_MMA": "bf16",
                          "VLLM_G128_PREFILL_CDNA2": "1",
                          "VLLM_G128_PREFILL_CDNA2_TILE": "64"},
    "gluon_cdna2_decode": {"VLLM_UA_TILE": "32", "VLLM_G128_GLUON": "64",
                          "VLLM_G128_GLUON_MMA": "bf16",
                          "VLLM_G128_DECODE_CDNA2": "1"},
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=49152)
    parser.add_argument("--query", type=int, default=512)
    parser.add_argument("--arm", action="append", choices=list(ARMS))
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--target-splits", type=int, choices=(8, 16, 32, 64),
                        help="Experimental fixed TP1 CDNA2 target split count")
    parser.add_argument("--segment-capacity", type=int, choices=(16, 64), default=16,
                        help="Use 64 to match production target scratch buffers")
    parser.add_argument("--kv-heads", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--query-tile", type=int, choices=(10, 16), default=10)
    args = parser.parse_args()
    if args.target_splits and (not args.split or args.kv_heads not in (2, 4)
                              or args.query > 16):
        parser.error("--target-splits needs --split, --kv-heads 2 or 4 and query <=16")
    if args.target_splits:
        os.environ["VLLM_MI210_FLASH_SPLITS"] = str(args.target_splits)
    else:
        os.environ.pop("VLLM_MI210_FLASH_SPLITS", None)
    import torch
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx90a"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.arm or ("generic", "gluon_bf16", "gluon_cdna2_bf16", "gluon_cdna2_tile64"):
        for key in CONTROLS:
            os.environ.pop(key, None)
        os.environ.update({"VLLM_G128_GLUON": "0", "VLLM_G128_PREFILL_GLUON": "0",
                           "VLLM_G128_DRAFT_GLUON": "0", "VLLM_UA_3D_MAXQ": "16",
                           **ARMS[name]})
        os.environ["VLLM_MI210_TP2_CDNA2"] = "1" if args.kv_heads == 2 else "0"
        os.environ["VLLM_MI210_G128_QUERY_TILE"] = str(args.query_tile)
        row = {"arm": name, "env": ARMS[name], "context": args.context,
               "query": args.query, "block_size": 1728, "split": args.split, "kv_heads": args.kv_heads, "heads": args.kv_heads * 6}
        row["target_splits"] = args.target_splits
        row["query_tile"] = args.query_tile
        row["two_card_collectives_validated"] = False
        row["segment_capacity"] = 64 if args.target_splits else args.segment_capacity
        tracker = None
        module = None
        original = None
        if name.startswith("gluon_cdna2"):
            import importlib
            decode = "decode" in name
            module = importlib.import_module("vllm.v1.attention.ops.gfx90a_g128_gluon_" +
                                             ("decode" if decode else "prefill"))
            symbol = "g128_core_cdna2" if decode else "g128_prefill_core_cdna2"
            original = getattr(module, symbol)

            class Tracker:
                launches = 0

                def __getitem__(self, grid):
                    self.launches += 1
                    return original[grid]

            tracker = Tracker()
            setattr(module, symbol, tracker)
        try:
            row.update(check_attention(256, args.kv_heads * 6, args.kv_heads, args.query, True, args.split,
                                       args.context, 1728, benchmark_repeats=5,
                                       segment_capacity=row["segment_capacity"]))
            if tracker is not None:
                row["candidate_launches"] = tracker.launches
                if not tracker.launches:
                    raise RuntimeError("Candidate dispatch was not exercised")
            row["numeric"] = "PASS"
        except Exception as error:
            row.update(numeric="FAIL", error=f"{type(error).__name__}: {error}")
        finally:
            if module is not None:
                setattr(module, symbol, original)
        with args.output.open("a") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()
        if row["numeric"] != "PASS":
            raise SystemExit(1)
