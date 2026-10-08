"""Plan or run matched C1 generation measurements at actual long input depths."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def commands(args):
    helpers = Path(__file__).resolve().parent
    for depth in args.depths:
        prompt = args.output_dir / f"prompt-{depth}.txt"
        output = args.output_dir / f"depth-{depth}.jsonl"
        generate = [sys.executable, str(helpers / "make_long_prompt.py"),
                    "--tokens", str(depth), "--model", args.tokenizer,
                    "--output", str(prompt)]
        bench = [sys.executable, str(helpers / "benchmark.py"),
                 "--model", args.model, "--prompt-file", str(prompt),
                 "--concurrency", "1", "--requests", "1",
                 "--max-tokens", str(args.max_tokens),
                 "--repetitions", str(args.repetitions),
                 "--api-key-env", args.api_key_env, "--output", str(output)]
        for url, label in zip(args.url, args.label):
            bench.extend(["--url", url, "--label", label])
        if args.image_id:
            bench.extend(["--image-id", args.image_id])
        yield depth, output, generate, bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--model", default="qwen-ptqr")
    parser.add_argument("--tokenizer", default="/models/Qwen3.8-27B-PTQR-R10S60")
    parser.add_argument("--depths", type=int, nargs="+",
                        default=[32768, 49152, 65536, 98304, 120000])
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--api-key-env", default="MI210_API_KEY")
    parser.add_argument("--image-id", help="Measured Docker image ID for artifact provenance")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="Without this option, print commands without sending requests")
    args = parser.parse_args()
    if len(args.url) != len(args.label) or len(set(args.label)) != len(args.label):
        parser.error("Each URL needs a distinct label")
    if min(args.max_tokens, args.repetitions, *args.depths) < 1:
        parser.error("Counts must be positive")
    # Reserve room for the chat template and retrieval instructions.
    if max(args.depths) + args.max_tokens + 512 > args.max_model_len:
        parser.error("Depth plus output and 512-token template reserve exceeds context limit")
    for depth, output, generate, bench in commands(args):
        print(json.dumps({"requested_background_tokens": depth,
                          "generate": generate, "benchmark": bench}), flush=True)
        if not args.execute:
            continue
        if output.exists():
            raise FileExistsError(f"Use a fresh output directory: {output}")
        subprocess.run(generate, check=True)
        subprocess.run(bench, check=True)
        rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
        if len(rows) != args.repetitions * len(args.url):
            raise RuntimeError(f"Incomplete campaign at depth {depth}")
        for row in rows:
            for sample in row["samples"]:
                if sample["usage"]["prompt_tokens"] < depth:
                    raise RuntimeError(f"Actual input is below requested depth {depth}")
                if sample["usage"]["completion_tokens"] != args.max_tokens:
                    raise RuntimeError(f"Output budget was not filled at depth {depth}")


if __name__ == "__main__":
    main()
