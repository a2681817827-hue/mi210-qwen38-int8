"""Compare post-conv GDN algorithms on the production TP1 C1 geometry."""

import argparse
import json
from pathlib import Path
import statistics

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.mamba.gdn.mi210_native_gdn import load_native_gdn
from vllm.third_party.flash_linear_attention.ops import fused_sigmoid_gating_delta_rule_update
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx90a')
    load_native_gdn()
    rows = []
    torch.manual_seed(741)
    for tokens in (8, 10, 14, 16):
        heads, value_heads, dim = 16, 48, 128
        mixed = torch.randn(tokens, 2 * heads * dim + value_heads * dim,
                            device='cuda', dtype=torch.bfloat16)
        q, k, v = mixed.split((heads * dim, heads * dim, value_heads * dim), dim=-1)
        q = q.view(1, tokens, heads, dim)
        k = k.view(1, tokens, heads, dim)
        v = v.view(1, tokens, value_heads, dim)
        b, a = torch.randn(tokens, 2 * value_heads, device='cuda', dtype=torch.bfloat16).chunk(2, dim=-1)
        log_a = torch.randn(value_heads, device='cuda') * 0.5
        bias = torch.randn(value_heads, device='cuda') * 0.1
        gate = torch.randn(tokens, value_heads, dim, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(dim, device='cuda', dtype=torch.bfloat16)
        initial = torch.randn(tokens + 1, value_heads, dim, dim, device='cuda') * 0.01
        baseline_state, native_state = initial.clone(), initial.clone()
        indices = torch.arange(1, tokens + 1, dtype=torch.int32, device='cuda').view(1, tokens)
        cu = torch.tensor([0, tokens], dtype=torch.int32, device='cuda')
        accepted = torch.tensor([4], dtype=torch.int32, device='cuda')
        out = torch.empty_like(gate)

        def baseline():
            raw, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=log_a, a=a, b=b, dt_bias=bias, q=q, k=k, v=v,
                initial_state=baseline_state, inplace_final_state=True,
                cu_seqlens=cu, ssm_state_indices=indices,
                num_accepted_tokens=accepted, scale=dim**-0.5,
                use_qk_l2norm_in_kernel=True)
            return rmsnorm_fn(raw.squeeze(0), weight, None, z=gate, eps=1e-6,
                              norm_before_gate=True, activation='silu')

        def native():
            return ops.fused_gdn_decode_post_conv_mtp(
                mixed_qkv=mixed, a=a, b=b, A_log=log_a, dt_bias=bias,
                state_indices=indices, cu_seqlens=cu, num_accepted_tokens=accepted,
                state=native_state, output_gate=gate, norm_weight=weight,
                out=out, scale=dim**-0.5, norm_eps=1e-6)

        graphs, outputs = [], []
        for operation in (baseline, native):
            for _ in range(3):
                operation()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = operation()
            graphs.append(graph)
            outputs.append(result)
        timings = [[], []]
        for repetition in range(5):
            # Reverse order so a single ordering cannot favor the candidate.
            for index in ((0, 1) if repetition % 2 == 0 else (1, 0)):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    graphs[index].replay()
                end.record()
                end.synchronize()
                timings[index].append(start.elapsed_time(end) / 50)
        relative = ((outputs[1].float() - outputs[0].float()).norm() /
                    outputs[0].float().norm().clamp_min(1e-20)).item()
        state_relative = ((native_state - baseline_state).norm() /
                          baseline_state.norm().clamp_min(1e-20)).item()
        assert relative < 0.0005, relative
        assert state_relative < 0.003, state_relative
        old, new = map(statistics.median, timings)
        row = {'tokens': tokens, 'reference_ms': old, 'native_ms': new,
               'speedup': old / new, 'relative_l2': relative,
               'state_relative_l2': state_relative, 'reference_raw_ms': timings[0],
               'native_raw_ms': timings[1], 'numeric': 'PASS', 'graph_replay': 'PASS'}
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
