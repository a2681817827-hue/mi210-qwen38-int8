"""Summarize leaf GPU kernels without counting graph annotations twice."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def audit(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as file:
        trace = json.load(file)
    groups = defaultdict(lambda: {"calls": 0, "duration_us": 0.0})
    intervals = defaultdict(list)
    shapes = defaultdict(int)
    for event in trace["traceEvents"]:
        if event.get("ph") != "X":
            continue
        if event.get("cat") == "kernel":
            start, duration = event["ts"], event["dur"]
            if not all(math.isfinite(x) for x in (start, duration)) or duration < 0:
                raise ValueError("Invalid kernel timing")
            group = groups[event["name"]]
            group["calls"] += 1
            group["duration_us"] += duration
            device = str(event.get("args", {}).get("device", event.get("pid")))
            intervals[device].append((start, start + duration))
        if event.get("cat") == "cpu_op" and "gemm" in event["name"].lower():
            dims = event.get("args", {}).get("Input Dims", [])
            if len(dims) >= 2 and len(dims[0]) == len(dims[1]) == 2:
                shapes[(event["name"], tuple(dims[0]), tuple(dims[1]))] += 1
    total = sum(group["duration_us"] for group in groups.values())
    if total <= 0:
        raise ValueError("No timed leaf kernels found")
    devices = {}
    for device, ranges in intervals.items():
        ranges.sort()
        first, end = ranges[0]
        busy, start = 0.0, first
        for left, right in ranges[1:]:
            if left > end:
                busy += end - start
                start = left
            end = max(end, right)
        busy += end - start
        # The largest end may precede the last start when streams overlap.
        span = max(right for _, right in ranges) - first
        devices[device] = {"busy_union_ms": busy / 1000,
                           "first_to_last_kernel_ms": span / 1000}
    kernels = []
    for name, group in sorted(groups.items(),
                              key=lambda item: item[1]["duration_us"], reverse=True):
        kernels.append({"name": name, "calls": group["calls"],
                        "total_ms": group["duration_us"] / 1000,
                        "mean_us": group["duration_us"] / group["calls"],
                        "fraction_of_summed_kernel_time": group["duration_us"] / total})
    return {"trace": str(path), "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "analysis_uses_gpu": False,
            "measurement": "leaf kernel durations; graph and CPU annotations excluded",
            "limitation": "Kernel duration shares are not end-to-end speedup predictions; "
                          "overlapping streams count separately in the summed denominator. "
                          "CPU shapes can be absent for captured GEMMs.",
            "summed_kernel_ms": total / 1000, "devices": devices, "kernels": kernels,
            "recorded_gemm_inputs": [
                {"op": op, "first_input": a, "second_input": b, "calls": count}
                for (op, a, b), count in sorted(shapes.items())]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({"report": str(args.output), "summed_kernel_ms": report["summed_kernel_ms"],
                      "top_kernels": report["kernels"][:5]}, indent=2))
