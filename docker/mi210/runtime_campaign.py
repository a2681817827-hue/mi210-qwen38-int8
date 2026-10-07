"""Measure an existing MI210 service without exposing its API credential."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[32768, 48000, 65536])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--short", action="store_true",
                        help="Benchmark the short-input C1 goal instead of long-context depths")
    parser.add_argument("--requests", type=int, default=4,
                        help="Sequential requests per repetition for --short")
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    if min(args.repetitions, args.max_tokens, args.requests) < 1:
        parser.error("repetitions, max-tokens and requests must be positive")
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    argv = Path("/proc/1/cmdline").read_bytes().decode().split("\0")

    def option(key, default=None):
        return argv[argv.index(key) + 1] if key in argv else default

    port = option("--port", "8000")
    host = option("--host", "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    url = f"http://{host}:{port}"
    key = option("--api-key")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if key:
        env["MI210_API_KEY"] = key
    args.output_dir.mkdir(parents=True)
    helpers = Path(__file__).resolve().parent
    model = option("--served-model-name", "qwen-ptqr")
    tokenizer = "/models/Qwen3.8-27B-PTQR-R10S60"
    spec = json.loads(option("--speculative-config", "{}"))
    metadata = {"label": args.label, "image_id": args.image_id,
                "speculative": spec, "max_model_len": option("--max-model-len"),
                "workload": "short-C1" if args.short else "long-C1",
                "primary_metric": "wall_output_tokens_per_second" if args.short
                else "streaming_generation_estimate_excluding_TTFT",
                "minimum_short_c1_wall_tokens_per_second": 92 if args.short else None,
                "target_short_c1_wall_tokens_per_second": 120 if args.short else None,
                "target_long_c1_generation_tokens_per_second": 120 if not args.short else None,
                "operator_env": {k: v for k, v in os.environ.items()
                                 if k.startswith(("VLLM_", "HSA_", "HIP_", "AITER_"))
                                 and not any(s in k for s in ("KEY", "TOKEN", "SECRET"))},
                "quality": "NOT_RUN", "status": "running"}
    meta_path = args.output_dir / "runtime.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with httpx.Client(headers=headers, timeout=1800) as client:
        client.get(url + "/health").raise_for_status()
        for depth in ([0] if args.short else args.depths):
            suffix = "short" if args.short else str(depth)
            prompt = args.output_dir / f"prompt-{suffix}.txt"
            output = args.output_dir / f"depth-{suffix}.jsonl"
            if args.short:
                prompt.write_text("Explain how matrix multiplication works, with examples.\n")
            else:
                subprocess.run([sys.executable, str(helpers / "make_long_prompt.py"),
                            "--tokens", str(depth), "--model", tokenizer,
                            "--output", str(prompt)], env=env, check=True)
            (args.output_dir / f"metrics-{suffix}-before.prom").write_text(
                client.get(url + "/metrics").text)
            if args.profile_only:
                client.post(url + "/start_profile").raise_for_status()
            try:
                subprocess.run([sys.executable, str(helpers / "benchmark.py"),
                                "--url", url, "--label", args.label, "--model", model,
                                "--prompt-file", str(prompt), "--concurrency", "1",
                                "--requests", str(args.requests if args.short else 1),
                                "--max-tokens", str(args.max_tokens),
                                "--repetitions", str(args.repetitions),
                                "--image-id", args.image_id, "--output", str(output)],
                               env=env, check=True)
            finally:
                if args.profile_only:
                    client.post(url + "/stop_profile").raise_for_status()
            (args.output_dir / f"metrics-{suffix}-after.prom").write_text(
                client.get(url + "/metrics").text)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            if len(rows) != args.repetitions:
                raise RuntimeError("Incomplete benchmark")
            for row in rows:
                if len(row["samples"]) != (args.requests if args.short else 1):
                    raise RuntimeError("Incomplete request batch")
                for sample in row["samples"]:
                    usage = sample["usage"]
                    if args.short and not 0 < usage["prompt_tokens"] <= 128:
                        raise RuntimeError("Short-input benchmark must use 1 to 128 prompt tokens")
                    if usage["prompt_tokens"] < depth or usage["completion_tokens"] != args.max_tokens:
                        raise RuntimeError("Input depth or output budget was not satisfied")
        metadata["status"] = "complete"
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
