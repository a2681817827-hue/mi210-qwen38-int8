"""Audit recorded streaming estimates without mixing input depths or concurrency."""
import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(paths, target=80.0):
    groups = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["concurrency"] != 1 or len(row["samples"]) != 1:
                raise ValueError(f"{path}:{line_number}: expected one C1 request")
            sample = row["samples"][0]
            usage = sample["usage"]
            tpot = sample["tpot_estimate_seconds"]
            wall = row["wall_output_tokens_per_second"]
            ttft = sample["ttft_seconds"]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                       for v in (tpot, wall, ttft)):
                raise ValueError(f"{path}:{line_number}: invalid timing")
            if usage["prompt_tokens"] < 32768 or usage["completion_tokens"] < 2:
                raise ValueError(f"{path}:{line_number}: insufficient input/output tokens")
            key = (row["label"], row["model"], row.get("image_id"),
                   usage["prompt_tokens"], usage["completion_tokens"],
                   row.get("timing_schema", "stream_end_v1"))
            group = groups.setdefault(key, {"decode": [], "ttft": [], "wall": [],
                                            "sources": [], "repetitions": set()})
            if row["repetition"] in group["repetitions"]:
                raise ValueError(f"Duplicate repetition for {key}")
            group["repetitions"].add(row["repetition"])
            group["decode"].append(1 / tpot)
            group["ttft"].append(ttft)
            group["wall"].append(wall)
            group["sources"].append(f"{path.name}:{line_number}")
    results = []
    for key, group in groups.items():
        label, model, image, depth, output, timing_schema = key
        median = statistics.median(group["decode"])
        results.append({"label": label, "model": model, "image_id": image,
                        "timing_schema": timing_schema,
                        "actual_input_tokens": depth, "output_tokens": output,
                        "samples": len(group["decode"]),
                        "generation_estimate_tokens_per_second": group["decode"],
                        "median_generation_estimate": median,
                        "minimum_generation_estimate": min(group["decode"]),
                        "ttft_seconds_in_record_order": group["ttft"],
                        "wall_output_tokens_per_second": group["wall"],
                        "target_tokens_per_second": target,
                        "median_target_met": median >= target,
                        "all_samples_target_met": min(group["decode"]) >= target,
                        "sources": group["sources"]})
    if not results:
        raise ValueError("No measurements found")
    return {"measurement": "recorded C1 streaming generation estimate, excludes TTFT",
            "summary_execution_uses_gpu": False,
            "quality_scope": "These timings do not establish model fidelity",
            "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--target", type=float, default=80.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.target) or args.target <= 0:
        parser.error("Target must be finite and positive")
    report = summarize(args.inputs, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
