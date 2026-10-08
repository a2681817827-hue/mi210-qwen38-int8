# SPDX-License-Identifier: Apache-2.0
"""Grouped INT8 target/draft attention against an FP32 reference on gfx90a."""

import os


def check_attention(head_size, heads, kv_heads, query_len, causal, split,
                    seq_len=1024, block_size=64, benchmark_repeats=0,
                    segment_capacity=16, benchmark_mode="eager"):
    import torch

    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    torch.manual_seed(17)
    blocks = (seq_len + block_size - 1) // block_size
    groups = head_size // 128
    # Match the serving cache's inline FP16 scale bytes and non-dense strides.
    pad = head_size + 2 * groups
    packed = torch.empty((blocks, kv_heads, block_size, 2 * pad),
                         dtype=torch.int8, device="cuda")
    base = torch.tensor([], dtype=torch.float16, device="cuda").set_(
        packed.untyped_storage())
    scale_shape = (blocks, block_size, kv_heads, groups)
    scale_stride = (packed.stride(0) // 2, pad, packed.stride(1) // 2, 1)
    ks = torch.as_strided(base, scale_shape, scale_stride,
                         storage_offset=head_size // 2)
    vs = torch.as_strided(base, scale_shape, scale_stride,
                         storage_offset=(pad + head_size) // 2)
    k = packed.transpose(1, 2)[..., :head_size]
    v = packed.transpose(1, 2)[..., pad:pad + head_size]
    k.random_(-64, 65)
    v.random_(-64, 65)
    ks.uniform_(0.003, 0.02)
    vs.uniform_(0.003, 0.02)
    q = torch.randn(query_len, heads, head_size, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    # Reverse physical page order to catch accidental linear-cache reads.
    physical = torch.arange(blocks - 1, -1, -1, device="cuda", dtype=torch.int32)
    table = physical.unsqueeze(0)
    segments = {}
    if split:
        segments = {
            "num_par_softmax_segments": segment_capacity,
            "softmax_segm_output": torch.empty((query_len, heads, segment_capacity, head_size),
                                               device="cuda"),
            "softmax_segm_max": torch.empty((query_len, heads, segment_capacity), device="cuda"),
            "softmax_segm_expsum": torch.empty((query_len, heads, segment_capacity), device="cuda"),
            "seq_threshold_3D": 1,
            "max_flash_decoding_splits": segment_capacity,
        }
    window = (-1, -1) if causal else (127, 127)
    args = (q, k, v, out,
        torch.tensor([0, query_len], dtype=torch.int32, device="cuda"),
        query_len, torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
        seq_len, head_size**-0.5, causal, window, table, 0.0,
        None, None, None)
    kwargs = dict(kv_quant_mode=KVQuantMode.INT8_BLOCK_G128,
                  g8_k_scale=ks, g8_v_scale=vs, **segments)
    unified_attention(*args, **kwargs)
    key = (k.float() * ks.float().repeat_interleave(128, -1))[physical.long()]
    value = (v.float() * vs.float().repeat_interleave(128, -1))[physical.long()]
    key = key.reshape(-1, kv_heads, head_size)[:seq_len].repeat_interleave(heads // kv_heads, 1)
    value = value.reshape(-1, kv_heads, head_size)[:seq_len].repeat_interleave(heads // kv_heads, 1)
    logits = torch.einsum("qhd,khd->hqk", q.float(), key) * head_size**-0.5
    qp = seq_len - query_len + torch.arange(query_len, device="cuda")
    kp = torch.arange(seq_len, device="cuda")
    if causal:
        mask = kp[None, :] <= qp[:, None]
    else:
        mask = (kp[None, :] >= qp[:, None] - 127) & (kp[None, :] <= qp[:, None] + 127)
    logits.masked_fill_(~mask[None], -float("inf"))
    expected = torch.einsum("hqk,khd->qhd", logits.softmax(-1), value)
    torch.testing.assert_close(out.float(), expected, atol=0.015, rtol=0.015)
    torch.cuda.synchronize()
    difference = out.float() - expected
    relative_l2 = (torch.linalg.vector_norm(difference) /
                   torch.linalg.vector_norm(expected).clamp_min(1e-12)).item()
    assert relative_l2 < 0.01, f"Relative L2 error {relative_l2} exceeds 1%"
    result = {"max_abs_error": difference.abs().max().item(),
              "relative_l2_error": relative_l2}
    if benchmark_repeats:
        # References are complete before timing; avoid counting compilation.
        import statistics
        for _ in range(2):
            unified_attention(*args, **kwargs)
        torch.cuda.synchronize()
        if benchmark_mode not in {"eager", "graph"}:
            raise ValueError("benchmark_mode must be eager or graph")
        run = lambda: unified_attention(*args, **kwargs)
        inner_repeats = 1
        if benchmark_mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            run = graph.replay
            inner_repeats = 10
            out.fill_(float("nan"))
            run()
            torch.cuda.synchronize()
            torch.testing.assert_close(out.float(), expected, atol=0.015, rtol=0.015)
        times = []
        for _ in range(benchmark_repeats):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(inner_repeats):
                run()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / inner_repeats)
        result.update(raw_ms=times, median_ms=statistics.median(times),
                      benchmark_mode=benchmark_mode)
    return result


def run(long_context=False):
    import torch

    assert torch.cuda.is_available(), "No GPU available"
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx90a"
    for key in ("VLLM_G128_GLUON", "VLLM_G128_PREFILL_GLUON", "VLLM_G128_DRAFT_GLUON"):
        os.environ[key] = "0"
    os.environ["VLLM_UA_3D_MAXQ"] = "16"
    cases = [(256, 6, 1, n, True, split)
             for n in (1, 7, 14, 256) for split in (False, True) if n <= 16 or not split]
    cases += [(128, 8, 2, n, False, False) for n in (7, 14)]
    for case in cases:
        check_attention(*case)
    count = len(cases)
    if long_context:
        # Actual hybrid-model serving pages are 1728 tokens, including a
        # partially filled final page at 32k. Cover decode and NS13 verify.
        for query_len in (1, 14):
            for split in (False, True):
                check_attention(256, 6, 1, query_len, True, split,
                                seq_len=32768, block_size=1728)
                count += 1
        check_attention(128, 8, 2, 14, False, False,
                        seq_len=32768, block_size=1728)
        count += 1
    return count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-context", action="store_true")
    args = parser.parse_args()
    print(f"grouped INT8 attention: {run(args.long_context)} cases PASS")
