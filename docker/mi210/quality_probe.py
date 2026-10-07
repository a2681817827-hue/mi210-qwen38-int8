"""Deterministic answer probes and baseline agreement; not a downstream eval."""

import argparse
import json
import os
from pathlib import Path

import httpx


def probes():
    rows = []
    for i in range(16):
        a, b = 17 + 3 * i, 25 + 2 * i
        rows.append((f"add-{i}", f"Return only the result of {a} + {b}.", str(a + b)))
        rows.append((f"mul-{i}", f"Return only the result of {i + 2} * {i + 3}.",
                     str((i + 2) * (i + 3))))
    expressions = [("len([2,4,6])", "3"), ("sum([1,3,5])", "9"),
                   ("7 // 2", "3"), ("17 % 5", "2"), ("2 ** 5", "32"),
                   ("max([3,7,1])", "7"), ("min([3,7,1])", "1"),
                   ("len('matrix')", "6")]
    rows.extend((f"python-{i}", f"Evaluate this Python expression. Return only its result: {expr}", ans)
                for i, (expr, ans) in enumerate(expressions))
    for i in range(6):
        code = f"MI210-{7319 + 11 * i}"
        rows.append((f"retrieve-{i}", f"The record's code is {code}. "
                     "Ignore this unrelated example: OTHER-0000. Return only the record's code.", code))
    facts = [("capital of France", "Paris"), ("capital of Japan", "Tokyo"),
             ("chemical symbol for oxygen", "O"), ("chemical symbol for gold", "Au"),
             ("number of sides of a triangle", "3"), ("number of days in a week", "7")]
    rows.extend((f"fact-{i}", f"What is the {question}? Return only the answer.", answer)
                for i, (question, answer) in enumerate(facts))
    return rows


def normalize(text):
    return text.strip().strip("`\"' .\n").casefold()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    reference = None
    if args.reference:
        reference = {r["id"]: r["output"] for r in json.loads(args.reference.read_text())["records"]}
    key = os.environ.get("MI210_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    rows = []
    with httpx.Client(headers=headers, timeout=1800) as client:
        for name, prompt, expected in probes():
            response = client.post(args.url.rstrip("/") + "/v1/chat/completions", json={
                "model": "qwen-ptqr", "temperature": 0, "max_tokens": 128,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": prompt}]})
            response.raise_for_status()
            body = response.json()
            output = body["choices"][0]["message"].get("content") or ""
            rows.append({"id": name, "expected": expected, "output": output,
                         "passed": normalize(output) == normalize(expected),
                         "exact_baseline_agreement": None if reference is None
                         else output == reference[name], "usage": body["usage"]})
    report = {"scope": "52 constrained greedy probes; not full model fidelity",
              "passed": sum(r["passed"] for r in rows), "total": len(rows),
              "exact_baseline_agreement": None if reference is None else
              sum(r["exact_baseline_agreement"] for r in rows), "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)
    if report["passed"] != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
