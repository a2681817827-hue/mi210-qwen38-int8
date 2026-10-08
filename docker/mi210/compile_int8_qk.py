"""Compile the optional INT8-QK Gluon kernel offline for gfx90a."""
import argparse
import hashlib
import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource
from triton.experimental.gluon import language as gl
from vllm.v1.attention.ops.gfx90a_g128_gluon_decode import g128_core_cdna2


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
signature = dict(Q='*bf16', K='*i8', V='*i8', SK='*fp16', SV='*fp16', BT='*i32',
                 SEQLENS='*i32', CUQ='*i32', PARTIAL='*fp32', PMAX='*fp32', PSUM='*fp32')
constants = dict(SCALE=0.0625, NUM_SEQS=1, NUM_QHEADS=24, NQ_PER_KV=6,
                 BLOCK_SIZE=1728, SPLITS=16, BT_STRIDE=64, Q_STRIDE0=6144,
                 Q_STRIDE1=256, K_STRIDE0=4*1728*520, K_STRIDE1=520,
                 K_STRIDE2=1728*520, V_STRIDE0=4*1728*520, V_STRIDE1=520,
                 V_STRIDE2=1728*520, S_STRIDE0=4*1728*260, S_STRIDE1=260,
                 S_STRIDE2=1728*260, MMA_DT=gl.bfloat16, MMA_FP16=False,
                 KV_TILE=64, Q_ROWS=64, Q_TOKENS=10, QK_INT8=True)
compiled = triton.compile(GluonASTSource(g128_core_cdna2, signature, constants),
                          target=GPUTarget('hip', 'gfx90a', 64), options={'num_warps':4})
assembly = compiled.asm['amdgcn']
instructions = sorted(set(line.strip() for line in assembly.splitlines() if 'mfma_i32' in line))
assert instructions and all('16x16x16' in line and 'i8' in line for line in instructions)
args.output.parent.mkdir(parents=True, exist_ok=True)
assembly_path = args.output.with_suffix('.amdgcn')
assembly_path.write_text(assembly)
args.output.write_text(json.dumps({'status':'OFFLINE_COMPILE_PASS', 'uses_gpu':False,
    'arch':'gfx90a', 'int8_mfma_instructions':instructions,
    'assembly_sha256':hashlib.sha256(assembly.encode()).hexdigest(),
    'numeric_validation':'NOT_RUN', 'constants':{k:str(v) for k,v in constants.items()}},indent=2)+'\n')
print(json.dumps({'status':'OFFLINE_COMPILE_PASS','uses_gpu':False,
                  'int8_mfma_instruction_count':len(instructions)}))
