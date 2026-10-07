"""Text, image and optional long-context serving smoke; not a quality eval."""
import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="qwen-ptqr")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--long-tokens", type=int, default=30000)
    args = parser.parse_args()
    if args.long_tokens < 1:
        parser.error("--long-tokens must be positive")
    headers = {"Authorization": "Bearer " + os.environ["MI210_API_KEY"]}
    probes = [
        ("arithmetic", "Return only the result of 17 + 25.", "42"),
        ("capital", "What is the capital of France? Answer briefly.", "paris"),
    ]
    from PIL import Image
    image = Image.new("RGB", (224, 224), (255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    probes.append(("vision", [
        {"type": "text", "text": "What color fills this image? Answer with one English color name."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                               base64.b64encode(buffer.getvalue()).decode()}},
    ], "red"))
    if args.long_context:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3.8-27B-PTQR-R10S60",
                                                  local_files_only=True)
        tokens = tokenizer.encode("This line contains ordinary background text. " * max(6000, args.long_tokens // 4 + 1000),
                                  add_special_tokens=False)[:args.long_tokens]
        prompt = "Remember the secret code MI210-7391.\n" + tokenizer.decode(tokens)
        prompt += "\nWhat was the secret code? Answer briefly."
        probes.append((f"requested_{args.long_tokens}_token_retrieval", prompt, "mi210-7391"))
    records = []
    with httpx.Client(timeout=1800, headers=headers) as client:
        for name, content, expected in probes:
            start = time.perf_counter()
            response = client.post(args.url.rstrip("/") + "/v1/chat/completions", json={
                "model": args.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "max_tokens": 256,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            response.raise_for_status()
            body = response.json()
            output = body["choices"][0]["message"].get("content") or ""
            record = {"probe": name, "seconds": time.perf_counter() - start,
                      "usage": body["usage"], "output": output,
                      "expected_fragment_present": expected in output.lower(),
                      "scope": "smoke_only_not_fidelity_eval"}
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    if not all(r["expected_fragment_present"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
