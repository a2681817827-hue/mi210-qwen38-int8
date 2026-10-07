# SPDX-License-Identifier: Apache-2.0
"""Opt-in fixed split count for MI210 target verification experiments."""


def validate_mi210_splits(value: str, capacity: int) -> int:
    if value not in {"8", "16", "32", "64"}:
        raise ValueError("VLLM_MI210_FLASH_SPLITS must be 8, 16, 32 or 64")
    splits = int(value)
    if splits > capacity:
        raise ValueError(
            f"MI210 split count {splits} exceeds segment capacity {capacity}"
        )
    return splits


def validate_mi210_query_tile(value: str) -> int:
    """Native TP1 GQA6 tiles: 5 uses 32 rows, 10 uses 64, 16 uses 128."""
    if value not in {"5", "10", "16"}:
        raise ValueError("VLLM_MI210_G128_QUERY_TILE must be 5, 10 or 16")
    return int(value)
