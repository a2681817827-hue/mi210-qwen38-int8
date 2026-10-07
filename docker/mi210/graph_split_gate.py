"""MI210 target split-K ragged and short-capture/long-replay numeric gate."""
import argparse
import json
import os
from pathlib import Path

import torch
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--splits', type=int, choices=(0, 8, 16, 32, 64), default=0,
                    help='0 tests the default heuristic')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--kv-tile', type=int, choices=(32, 64), default=64)
parser.add_argument('--block-size', type=int, default=1728)
parser.add_argument('--max-query', type=int, choices=(8, 10, 14, 16, 18), default=14)
parser.add_argument('--query-tile', type=int, choices=(5, 10, 16), default=10)
parser.add_argument('--kv-heads', type=int, choices=(2, 4), default=4,
                    help='2 exercises experimental TP2 12:2 geometry on one GPU')
parser.add_argument('--max-context', type=int, default=65537,
                    help='Extend numerical/graph qualification up to the 128K limit')
args = parser.parse_args()
if not 65537 <= args.max_context <= 131072:
    parser.error('max context must be in [65537, 131072]')
if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx90a':
    raise RuntimeError('This gate requires a gfx90a GPU')
os.environ.update({
    'VLLM_UA_TILE': '32', 'VLLM_G128_GLUON': '64',
    'VLLM_G128_GLUON_MMA': 'bf16', 'VLLM_G128_DECODE_CDNA2': '1',
    'VLLM_G128_DECODE_CDNA2_TILE': str(args.kv_tile),
    'VLLM_MI210_G128_QUERY_TILE': str(args.query_tile),
})
os.environ['VLLM_MI210_TP2_CDNA2'] = '1' if args.kv_heads == 2 else '0'
if args.splits:
    os.environ['VLLM_MI210_FLASH_SPLITS'] = str(args.splits)
else:
    os.environ.pop('VLLM_MI210_FLASH_SPLITS', None)
# Refuse a passing generic fallback: this gate must exercise the native core.
from vllm.v1.attention.ops import gfx90a_g128_gluon_decode as native
original_core = native.g128_core_cdna2
class LaunchTracker:
    launches = 0
    int8_qk_launches = 0
    def __getitem__(self, grid):
        launch = original_core[grid]
        def tracked_launch(*args, **kwargs):
            self.launches += 1
            if kwargs.get('QK_INT8') is True:
                self.int8_qk_launches += 1
            return launch(*args, **kwargs)
        return tracked_launch
tracker = LaunchTracker()
native.g128_core_cdna2 = tracker
os.environ['VLLM_UA_3D_MAXQ'] = '20' if args.max_query > 16 else '16'
torch.manual_seed(17)
block_size, heads, kvheads, dim, bound = args.block_size, args.kv_heads * 6, args.kv_heads, 256, args.max_context
if block_size < 64 or block_size % 64:
    parser.error('block size must be a positive multiple of the native tile64')
blocks = (bound + block_size - 1) // block_size
pad = dim + 4
packed = torch.empty((blocks, kvheads, block_size, 2 * pad), dtype=torch.int8, device='cuda')
base = torch.tensor([], dtype=torch.float16, device='cuda').set_(packed.untyped_storage())
strides = (packed.stride(0) // 2, pad, packed.stride(1) // 2, 1)
ks = torch.as_strided(base, (blocks, block_size, kvheads, 2), strides, storage_offset=dim // 2)
vs = torch.as_strided(base, (blocks, block_size, kvheads, 2), strides, storage_offset=(pad + dim) // 2)
k = packed.transpose(1, 2)[..., :dim]
v = packed.transpose(1, 2)[..., pad:pad + dim]
k.random_(-64, 65); v.random_(-64, 65)
ks.uniform_(0.003, 0.02); vs.uniform_(0.003, 0.02)
qlens = [args.max_query, 7, 1]
qstarts = [0, args.max_query, args.max_query + 7, args.max_query + 8]
q = torch.randn(qstarts[-1], heads, dim, device='cuda', dtype=torch.bfloat16)
out = torch.empty_like(q)
physical = torch.arange(blocks - 1, -1, -1, device='cuda', dtype=torch.int32)
table = physical.unsqueeze(0).repeat(3, 1)
cu = torch.tensor(qstarts, device='cuda', dtype=torch.int32)
lengths = torch.tensor([33, 4097, 33], device='cuda', dtype=torch.int32)
segments = dict(num_par_softmax_segments=64,
    softmax_segm_output=torch.empty((qstarts[-1], heads, 64, dim), device='cuda'),
    softmax_segm_max=torch.empty((qstarts[-1], heads, 64), device='cuda'),
    softmax_segm_expsum=torch.empty((qstarts[-1], heads, 64), device='cuda'),
    seq_threshold_3D=256, max_flash_decoding_splits=64)

def execute():
    unified_attention(q, k, v, out, cu, args.max_query, lengths, bound, dim**-0.5,
        True, (-1, -1), table, 0.0, None, None, None,
        kv_quant_mode=KVQuantMode.INT8_BLOCK_G128, g8_k_scale=ks, g8_v_scale=vs, **segments)

def reference(depths):
    key = (k.float() * ks.float().repeat_interleave(128, -1))[physical.long()].reshape(-1, kvheads, dim)
    value = (v.float() * vs.float().repeat_interleave(128, -1))[physical.long()].reshape(-1, kvheads, dim)
    answers = []
    for i, depth in enumerate(depths):
        kr = key[:depth].repeat_interleave(6, 1)
        vr = value[:depth].repeat_interleave(6, 1)
        qr = q[qstarts[i]:qstarts[i+1]].float()
        logits = torch.einsum('qhd,khd->hqk', qr, kr) * dim**-0.5
        qp = depth - qlens[i] + torch.arange(qlens[i], device='cuda')
        kp = torch.arange(depth, device='cuda')
        logits.masked_fill_(~(kp[None, :] <= qp[:, None])[None], -float('inf'))
        answers.append(torch.einsum('hqk,khd->qhd', logits.softmax(-1), vr))
    expected = torch.cat(answers)
    torch.testing.assert_close(out.float(), expected, atol=0.015, rtol=0.015)
    rel = (torch.linalg.vector_norm(out.float() - expected) /
           torch.linalg.vector_norm(expected).clamp_min(1e-12)).item()
    assert rel < 0.01, rel
    return rel

execute()
short_error = reference([33, 4097, 33])
if not tracker.launches:
    raise RuntimeError('Native CDNA2 core was not exercised')
if os.environ.get('VLLM_MI210_INT8_QK') == '1' and not tracker.int8_qk_launches:
    raise RuntimeError('INT8-QK specialization was not exercised')
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3): execute()
torch.cuda.current_stream().wait_stream(stream)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream): execute()
errors = []
middle = 48081 if bound == 65537 else min(120081, bound)
for depths in ([middle, bound, 33], [33, 4097, 33], [bound, middle, 4097]):
    lengths.copy_(torch.tensor(depths, device='cuda', dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    errors.append({'lengths': depths, 'relative_l2': reference(depths)})
row = {'splits': args.splits, 'query_tile': args.query_tile,
       'heads': heads, 'kv_heads': kvheads,
       'two_card_collectives_validated': False,
       'ragged_numeric': 'PASS', 'graph_replay': 'PASS',
       'block_size': block_size, 'query_lengths': qlens,
       'max_context': bound,
       'candidate_launches': tracker.launches,
       'int8_qk_launches': tracker.int8_qk_launches,
       'short_relative_l2': short_error, 'replays': errors}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('a') as file: file.write(json.dumps(row) + '\n')
print(json.dumps(row), flush=True)
