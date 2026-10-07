"""Interleave OpenAI streaming benchmarks; token counts come only from usage."""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx


async def request(client, url, model, prompt, max_tokens):
    start = time.perf_counter()
    first = None
    last = None
    usage = None
    content = []
    async with client.stream("POST", url.rstrip("/") + "/v1/chat/completions",
                             json={"model": model,
                                   "messages": [{"role": "user", "content": prompt}],
                                   "temperature": 0, "max_tokens": max_tokens,
                                   "stream": True,
                                   "stream_options": {"include_usage": True}}) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                text = choice.get("delta", {}).get("content") or ""
                if text:
                    last = time.perf_counter()
                    first = last if first is None else first
                    content.append(text)
    end = time.perf_counter()
    if not usage or not isinstance(usage.get("completion_tokens"), int):
        raise RuntimeError("Server did not return completion_tokens usage")
    tokens = usage["completion_tokens"]
    return {"usage": usage, "seconds": end - start,
            "ttft_seconds": None if first is None else first - start,
            "generation_seconds": None if last is None else last - first,
            "tpot_estimate_seconds": (last - first) / (tokens - 1)
            if first is not None and last > first and tokens > 1 else None,
            "output": "".join(content)}


async def run(args):
    prompts = (Path(args.prompt_file).read_text().splitlines() if args.prompt_file
               else ["Explain how matrix multiplication works, with examples."])
    if not prompts or any(not p.strip() for p in prompts):
        raise ValueError("Prompt file must contain nonempty prompts, one per line")
    labels = args.label or args.url
    if len(labels) != len(args.url):
        raise ValueError("Provide one label per URL")
    api_key = os.environ.get(args.api_key_env)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=1800, headers=headers) as client:
        for repetition in range(args.repetitions):
            order = list(range(len(args.url)))
            order = order[repetition % len(order):] + order[:repetition % len(order)]
            for index in order:
                semaphore = asyncio.Semaphore(args.concurrency)

                async def sample(number):
                    async with semaphore:
                        return await request(client, args.url[index], args.model,
                                             prompts[number % len(prompts)], args.max_tokens)

                start = time.perf_counter()
                samples = await asyncio.gather(*(sample(n) for n in range(args.requests)))
                elapsed = time.perf_counter() - start
                row = {"label": labels[index], "url": args.url[index],
                       "repetition": repetition, "model": args.model,
                       "image_id": args.image_id, "concurrency": args.concurrency,
                       "wall_seconds": elapsed,
                       "wall_output_tokens_per_second": sum(
                           s["usage"]["completion_tokens"] for s in samples) / elapsed,
                       "samples": samples, "quality_gate": "unvalidated",
                       "acceptance": None,
                       "timing_schema": "first_last_content_v2",
                       "tpot_note": "Estimate between first and last content SSE events; "
                       "speculative chunks can contain multiple tokens. "
                       "One content event cannot establish generation speed."}
                with Path(args.output).open("a") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k != "samples"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--label", action="append")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--image-id")
    parser.add_argument("--api-key-env", default="MI210_API_KEY",
                        help="Environment variable containing the API key")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.concurrency, args.requests, args.max_tokens, args.repetitions) < 1:
        parser.error("Counts must be positive")
    asyncio.run(run(args))
