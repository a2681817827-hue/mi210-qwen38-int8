"""Prepare long-input C1 comparisons; do not start or alter GPU services."""

import argparse
import json
from pathlib import Path

from short_c1_plan import candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=48000)
    parser.add_argument("--image", default="int8-vllm-mi210:short-c1-candidates")
    args = parser.parse_args()
    if not 1 <= args.depth <= 120000:
        parser.error("depth must be in [1, 120000]")
    rows = []
    for label, settings in candidates():
        name = "mi210-long-" + label
        rows.append({"label": label, "container": name,
                     "env": {**settings, "MI210_IMAGE": args.image,
                         "MI210_NAME": name, "MAX_MODEL_LEN": "131072",
                         "MI210_COMPILE_CACHE": "/data/cache/mi210-vllm/long-c1/" + label,
                         "MI210_TRITON_CACHE": "/data/cache/mi210-vllm/triton-cdna2"},
                     "launch_argv": ["bash", "scripts/serve_mi210_long_context.sh", "graph-c8"],
                     "benchmark_argv": ["/opt/int8-vllm/.venv/bin/python",
                         "/opt/int8-vllm/docker/mi210/runtime_campaign.py",
                         "--depths", str(args.depth), "--repetitions", "5",
                         "--max-tokens", "512", "--label", label,
                         "--image-id", "REPLACE_WITH_ACTUAL_IMAGE_ID",
                         "--output-dir", "/tmp/mi210-long/" + label],
                     "quality_gate_required": True})
    report = {"status": "PLANNED_NOT_EXECUTED", "priority": "long-input C1",
              "input_token_budget": args.depth, "client_concurrency": 1,
              "server_max_num_seqs": 8, "context_limit": 131072,
              "primary_metric": "streaming generation estimate excluding TTFT",
              "also_report": ["TTFT per repetition", "wall output tokens/s",
                              "draft acceptance", "actual API input/output token counts"],
              "target_generation_tokens_per_second": 120,
              "short_C1_regression_minimum_wall_tokens_per_second": 92,
              "hardware_validated": False,
              "instructions": "Run serially. Record cold and reused-prefix TTFT separately. "
              "Compare identical prompts and output budgets; reverse control/winner order. "
              "Do not treat C8 aggregate throughput as C1 generation speed.",
              "candidates": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({"plan": str(args.output), "candidates": len(rows),
                      "hardware_validated": False}))


if __name__ == "__main__":
    main()
