"""Reuse the repository GDN state/output reference test, including graph replay."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm import _custom_ops as ops
from vllm.third_party.flash_linear_attention.ops import fused_sigmoid_gating_delta_rule_update
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-index', type=int, choices=range(20),
                        help='Isolate one case for diagnosis; not a full qualification')
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx90a')
    assert not hasattr(torch.ops._C, 'fused_gdn_decode_post_conv_mtp')
    torch.ops.load_library('/opt/int8-vllm/docker/mi210/native-gdn-build/mi210_native_gdn.so')
    tree = ast.parse(args.reference.read_text())
    test = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'test_fused_gdn_decode_post_conv_mtp_head_ratios')
    test.decorator_list = []
    # A single-token view is contiguous by PyTorch's size-one rule, even
    # though its row stride retains the interleaved BA storage contract.
    # Test that contract directly without weakening any numeric assertions.
    for node in ast.walk(test):
        if isinstance(node, ast.Assert):
            for tensor in ('a', 'b'):
                if ast.unparse(node.test) == f'not {tensor}.is_contiguous()':
                    node.test = ast.parse(
                        f'{tensor}.stride(0) == 2 * HV', mode='eval').body
    test.body.extend(ast.parse('''state_l2 = ((state_actual.float() - state_ref.float()).norm() /
                  state_ref.float().norm().clamp_min(1e-20)).item()
assert state_l2 < 0.003, state_l2
gate_statistics.append(state_l2)
''').body)
    replays = []
    gate_statistics = []

    def native_with_replay(**kwargs):
        state = kwargs['state']
        initial = state.clone()
        torch.cuda.synchronize()
        print(json.dumps({'phase': 'reference_ready'}), flush=True)
        result = ops.fused_gdn_decode_post_conv_mtp(**kwargs)
        torch.cuda.synchronize()
        print(json.dumps({'phase': 'native_eager_complete'}), flush=True)
        expected_output, expected_state = result.clone(), state.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state.copy_(initial)
            ops.fused_gdn_decode_post_conv_mtp(**kwargs)
        for _ in range(3):
            result.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(result, expected_output, rtol=0, atol=0)
            torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        replays.append('PASS')
        return result

    def no_skip(reason):
        raise RuntimeError('A mandatory gate attempted to skip: ' + reason)

    namespace = dict(torch=torch, pytest=SimpleNamespace(skip=no_skip),
                     ops=SimpleNamespace(fused_gdn_decode_post_conv_mtp=native_with_replay),
                     fused_sigmoid_gating_delta_rule_update=fused_sigmoid_gating_delta_rule_update,
                     rmsnorm_fn=rmsnorm_fn, gate_statistics=gate_statistics)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[test], type_ignores=[])),
                 str(args.reference), 'exec'), namespace)
    rows = []
    case_index = 0
    for ratio, tp, lengths in ((3, 1, (1,)), (3, 1, (8,)), (3, 1, (14,)),
                               (3, 1, (16,)), (3, 1, (14, 7, 0)),
                               (3, 1, (14,) * 8), (1, 1, (16,)),
                               (2, 1, (14, 8, 0)), (4, 4, (16,)), (8, 4, (14,))):
        for norm in (torch.float32, torch.bfloat16):
            current_index = case_index
            case_index += 1
            if args.case_index is not None and args.case_index != current_index:
                continue
            print(json.dumps({'phase': 'case_start', 'case_index': current_index,
                              'head_ratio': ratio, 'query_lengths': lengths,
                              'norm_dtype': str(norm)}), flush=True)
            namespace[test.name](ratio, tp, lengths, torch.float32, norm)
            rows.append(dict(head_ratio=ratio, tp=tp, query_lengths=lengths,
                             state_dtype='float32', norm_dtype=str(norm), numeric='PASS',
                             output_relative_l2_threshold=0.0005,
                             state_relative_l2=gate_statistics[-1],
                             state_relative_l2_threshold=0.003,
                             state_tolerance={'atol':0.03,'rtol':0.03}, graph_replay='PASS'))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'status':'RUNNING','cases':rows,
                                              'stateful_replay_checks':len(replays)},indent=2)+'\n')
            print(json.dumps(rows[-1]), flush=True)
    args.output.write_text(json.dumps({'status':'PASS' if args.case_index is None else 'DIAGNOSTIC_PASS',
                                      'cases':rows,
                                      'stateful_replay_checks':len(replays)},indent=2)+'\n')


if __name__ == '__main__':
    main()
