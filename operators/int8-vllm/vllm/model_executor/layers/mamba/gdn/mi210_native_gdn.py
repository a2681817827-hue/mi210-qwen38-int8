# SPDX-License-Identifier: Apache-2.0
"""Strict opt-in loader for the separately built MI210 GDN experiment."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_native_gdn() -> None:
    from vllm.platforms.rocm import on_gfx90a

    if not on_gfx90a():
        raise RuntimeError('The native MI210 GDN experiment requires gfx90a')
    build = Path('/opt/int8-vllm/docker/mi210/native-gdn-build')
    provenance = json.loads((build / 'provenance.json').read_text())
    if provenance['arch'] != 'gfx90a' or provenance['max_verify_tokens'] != 16:
        raise RuntimeError('Native GDN provenance does not match this experiment')
    source = build / 'mi210_gdn.cu'
    if hashlib.sha256(source.read_bytes()).hexdigest() != provenance['generated_sha256']:
        raise RuntimeError('Native GDN generated source drift')
    if hasattr(torch.ops._C, 'fused_gdn_decode_post_conv_mtp'):
        raise RuntimeError('A different fused GDN operator is already registered')
    torch.ops.load_library(str(build / 'mi210_native_gdn.so'))
