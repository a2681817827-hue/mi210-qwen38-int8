"""Interleave captured C1/48K attention timings, preserving G128 cache dtype."""
import json
import os
from pathlib import Path
import statistics

import torch
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

import hashlib
from vllm.v1.attention.ops import gfx90a_g128_gluon_qkpv as candidate
original = candidate.g128_core_cdna2
warps, warp_n = 0, 1
resources = {}
calls = 0
class CountCandidate:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            global calls
            result = original[grid](*args, **kwargs)
            calls += 1
            resources["candidate"] = {"n_regs": getattr(result, "n_regs", None),
                                      "n_spills": getattr(result, "n_spills", None)}
            return result
        return launch
candidate.g128_core_cdna2 = CountCandidate()

output = Path('/results/qkpv-serving-dispatch-gate.json')
if output.exists():
    raise RuntimeError('Use a fresh microbenchmark result.')
os.environ.update(VLLM_UA_TILE='32', VLLM_G128_GLUON='64',
                  VLLM_G128_GLUON_MMA='bf16', VLLM_G128_DECODE_CDNA2='1',
                  VLLM_G128_DECODE_CDNA2_TILE='64', VLLM_MI210_INT8_QK='1',
                  VLLM_UA_3D_MAXQ='16')
if torch.cuda.device_count() != 1 or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx90a':
    raise RuntimeError('One isolated MI210 required.')
torch.manual_seed(23)
depth, block_size, heads, kvheads, dim = 48081, 1728, 24, 4, 256
blocks = (131072 + block_size - 1) // block_size
packed = torch.empty((blocks, kvheads, block_size, 2 * (dim + 4)), device='cuda', dtype=torch.int8)
base = torch.tensor([], dtype=torch.float16, device='cuda').set_(packed.untyped_storage())
strides = (packed.stride(0) // 2, dim + 4, packed.stride(1) // 2, 1)
ks = torch.as_strided(base, (blocks, block_size, kvheads, 2), strides, storage_offset=dim // 2)
vs = torch.as_strided(base, (blocks, block_size, kvheads, 2), strides, storage_offset=(dim + 4 + dim) // 2)
k = packed.transpose(1, 2)[..., :dim]
v = packed.transpose(1, 2)[..., dim + 4:dim + 4 + dim]
k.random_(-64, 65); v.random_(-64, 65)
ks.uniform_(0.003, 0.02); vs.uniform_(0.003, 0.02)
q = torch.randn((14, heads, dim), device='cuda', dtype=torch.bfloat16)
out = torch.empty_like(q)
table = torch.arange(blocks - 1, -1, -1, device='cuda', dtype=torch.int32).unsqueeze(0)
cu = torch.tensor([0, 14], device='cuda', dtype=torch.int32)
lengths = torch.tensor([depth], device='cuda', dtype=torch.int32)
segments = dict(num_par_softmax_segments=64,
    softmax_segm_output=torch.empty((14, heads, 64, dim), device='cuda'),
    softmax_segm_max=torch.empty((14, heads, 64), device='cuda'),
    softmax_segm_expsum=torch.empty((14, heads, 64), device='cuda'),
    seq_threshold_3D=256, max_flash_decoding_splits=64)
def run():
    os.environ["VLLM_MI210_QKPV_INTERLEAVE"] = "1" if warps else "0"
    unified_attention(q, k, v, out, cu, 14, lengths, 131072, dim**-0.5,
        True, (-1, -1), table, 0.0, None, None, None,
        kv_quant_mode=KVQuantMode.INT8_BLOCK_G128, g8_k_scale=ks, g8_v_scale=vs, **segments)
cases = [(0, 1), (2, 1)]
graphs = {}
reference = None
errors = {}
for warps, warp_n in cases:
    os.environ['VLLM_MI210_G128_QUERY_TILE'] = '5'
    os.environ['VLLM_G128_DECODE_CDNA2_TILE'] = '64'
    os.environ['VLLM_MI210_FLASH_SPLITS'] = '32'
    run()
    if reference is None:
        reference = out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream): run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference, rtol=0.008, atol=0.004)
    error = (torch.linalg.vector_norm(out.float() - reference.float()) /
             torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)).item()
    if not torch.equal(out, reference):
        raise RuntimeError('Geometry-only change exceeded relative L2 0.0005: ' + str(error))
    graphs[warps, warp_n] = graph
    errors[warps, warp_n] = error
timings = {case: [] for case in cases}
for round_number in range(7):
    # Reverse each alternate round to limit simple order/clock bias.
    order = cases if round_number % 2 == 0 else list(reversed(cases))
    for case in order:
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(50): graphs[case].replay()
        end.record(); end.synchronize()
        timings[case].append(begin.elapsed_time(end) / 50)
baseline = statistics.median(timings[0, 1])
exact_replays=[]
for new_depth in (33, 48081, 131072, 120081):
    lengths.fill_(new_depth); q.normal_()
    out.fill_(float('nan')); graphs[0,1].replay(); torch.cuda.synchronize(); expected=out.clone()
    for case in cases[1:]:
        out.fill_(float('nan')); graphs[case].replay(); torch.cuda.synchronize()
        assert torch.isfinite(out).all() and torch.equal(out,expected),(new_depth,case)
        exact_replays.append({'depth':new_depth,'variant':case,'output_bit_exact':True})
rows = [{'warps': warps, 'warp_n': warp_n, 'median_ms': statistics.median(timings[warps, warp_n]),
         'raw_ms': timings[warps, warp_n], 'numeric': 'PASS', 'graph_replay': 'PASS',
         'relative_l2_to_control': errors[warps, warp_n],
         'operator_speedup': baseline / statistics.median(timings[warps, warp_n])}
        for warps, warp_n in cases]
if calls == 0:
    raise RuntimeError("Candidate dispatch was never invoked")
report = {"actual_candidate_calls": calls, "serving_validated": False, 'status': 'PASS', 'exact_replays':exact_replays, 'kv_tile':64, 'splits':32, 'query_tile':5, 'source_sha256':hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(), 'resources':resources, 'input_tokens': depth, 'query_tokens': 14, 'client_concurrency': 1,
          'measurement': 'captured attention operator only; not model token/s', 'cases': rows}
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
