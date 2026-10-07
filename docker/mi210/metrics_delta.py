"""Speculative acceptance from Prometheus counter deltas for one workload."""

from pathlib import Path


COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


def counters(text):
    result = {key: 0.0 for key in COUNTERS}
    found = set()
    for line in text.splitlines():
        for key, name in COUNTERS.items():
            if line.startswith(name + "{") or line.startswith(name + " "):
                result[key] += float(line.rsplit(None, 1)[1])
                found.add(key)
    if found != set(COUNTERS):
        raise ValueError("Missing speculative counters")
    return result


def acceptance(before: Path, after: Path):
    a, b = counters(before.read_text()), counters(after.read_text())
    delta = {key: b[key] - a[key] for key in a}
    if any(value < 0 for value in delta.values()):
        raise ValueError("Counters reset during workload")
    if delta["accepted_tokens"] > delta["draft_tokens"]:
        raise ValueError("Accepted tokens exceed drafted tokens")
    return {**delta,
            "accepted_fraction": delta["accepted_tokens"] / delta["draft_tokens"]
            if delta["draft_tokens"] else None,
            "mean_accepted_per_draft": delta["accepted_tokens"] / delta["drafts"]
            if delta["drafts"] else None}
