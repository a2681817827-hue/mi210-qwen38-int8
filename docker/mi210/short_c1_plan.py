"""Create a serial short-input C1 experiment plan without starting containers."""

import argparse
import json
from pathlib import Path


def candidates():
    # Change one variable from the NS13 control before combining winners.
    control = {"SPEC_TOKENS": "13", "MI210_QUERY_TILE": "10",
               "MI210_INT8_LM_HEAD": "0"}
    yield "control-ns13", control
    for ns in (7, 9):
        yield f"ns{ns}", {**control, "SPEC_TOKENS": str(ns)}
    yield "query16", {**control, "MI210_QUERY_TILE": "16"}
    yield "split64", {**control, "MI210_FLASH_SPLITS": "64"}
    yield "int8-head", {**control, "MI210_INT8_LM_HEAD": "1"}
    yield "control-ns13-repeat", control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="int8-vllm-mi210:short-c1-candidates")
    args = parser.parse_args()
    rows = []
    for label, settings in candidates():
        name = "mi210-short-" + label
        env = {**settings, "MI210_IMAGE": args.image, "MI210_NAME": name,
               "MAX_MODEL_LEN": "131072",
               "MI210_COMPILE_CACHE": "/data/cache/mi210-vllm/short-c1/" + label,
               "MI210_TRITON_CACHE": "/data/cache/mi210-vllm/triton-cdna2"}
        rows.append({"label": label, "container": name, "env": env,
                     "launch_argv": ["bash", "scripts/serve_mi210_long_context.sh", "graph-c8"],
                     "benchmark_argv": ["/opt/int8-vllm/.venv/bin/python",
                         "/opt/int8-vllm/docker/mi210/runtime_campaign.py",
                         "--short", "--requests", "4", "--repetitions", "5",
                         "--max-tokens", "512", "--label", label,
                         "--image-id", "REPLACE_WITH_ACTUAL_IMAGE_ID",
                         "--output-dir", "/tmp/mi210-short/" + label],
                     "requires_numeric_and_model_quality_gate": label == "int8-head"})
    report = {"status": "PLANNED_NOT_EXECUTED", "primary_metric": "C1 wall output tokens/s",
              "minimum_wall_tokens_per_second": 92,
              "target": 120, "client_concurrency": 1, "server_max_num_seqs": 8,
              "context_limit": 131072, "hardware_validated": False,
              "instructions": "Run serially, stop each arm before launching the next; "
              "warm the server before measurement. Record image ID and acceptance metrics. "
              "Rerun winners and control with reversed order; run long-context regression.",
              "candidates": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({"plan": str(args.output), "candidates": len(rows),
                      "hardware_validated": False}))


if __name__ == "__main__":
    main()
