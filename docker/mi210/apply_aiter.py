# SPDX-License-Identifier: Apache-2.0
"""Adapt the fork's source AITER checkout to CDNA2 without importing AITER.

CK support follows davetha/aiter-cdna2 v1.6, adapted to int8-aiter's existing
dtype-pruning machinery. Optional BF16 ASM edits use that project's MIT patch
definitions verbatim; its LICENSE remains in aiter-cdna2-v1.6/.
"""

import argparse
import ast
import json
import runpy
from pathlib import Path


def replace_exact(body: str, old: str, new: str, count: int = 1) -> str:
    if body.count(new) == count and body.count(old) == 0:
        return body
    if body.count(old) != count:
        raise ValueError(f"Patch anchor changed: expected {count}: {old!r}")
    return body.replace(old, new)


def patch_checkout(root: Path, *, check: bool = False, asm: bool = False) -> dict:
    updates = {}
    targets = root / "aiter/jit/utils/build_targets.py"
    body = targets.read_text()
    tree = ast.parse(body)
    mapping = next(
        n.value for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "GFX_CU_NUM_MAP"
                for t in n.targets)
    )
    known = ast.literal_eval(mapping)
    if "gfx90a" not in known:
        # Some local images already carry an additive assignment.
        assignment = 'GFX_CU_NUM_MAP["gfx90a"] = 104'
        if assignment not in body:
            body = replace_exact(body, "GFX_CU_NUM_MAP = {\n",
                                 'GFX_CU_NUM_MAP = {\n    "gfx90a": 104,  # MI210\n')
    elif known["gfx90a"] != 104:
        raise ValueError("Unexpected gfx90a default CU count")
    updates[targets] = body

    gen = root / "csrc/ck_gemm_a8w8/gen_instances.py"
    body = replace_exact(
        gen.read_text(),
        'elif all(gfx == "gfx908" for gfx, _ in get_build_targets()):',
        'elif all(gfx in ("gfx908", "gfx90a") '
        'for gfx, _ in get_build_targets()):',
    )
    if "gen_built_combos_header" not in body:
        raise ValueError("AITER dtype-pruning manifest support is required")
    updates[gen] = body

    if asm:
        vendor = Path(__file__).parent / "aiter-cdna2-v1.6"
        patch = runpy.run_path(str(vendor / "patches/enable_gfx90a_asm_paths.py"))
        paths = {
            "mha.py": root / "aiter/ops/mha.py",
            "mha_fwd.cu": root / "csrc/cpp_itfs/mha_fwd.cu",
        }
        for path, old, new, count in patch["PATCHES"]:
            target = paths.get(Path(path).name)
            if target is not None:
                body = updates.get(target, target.read_text())
                updates[target] = replace_exact(body, old, patch["expand"](new), count)

    # Validate every anchor first so a drifted source never gets half-patched.
    changed = [str(p.relative_to(root)) for p, body in updates.items()
               if p.read_text() != body]
    if check and changed:
        raise ValueError(f"Unapplied MI210 patches: {changed}")
    if not check:
        for path, body in updates.items():
            path.write_text(body)
    return {"ck_int8": True, "bf16_asm_source": asm, "changed": changed,
            "hardware_validated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--asm", action="store_true")
    args = parser.parse_args()
    print(json.dumps(patch_checkout(args.root, check=args.check, asm=args.asm)))
