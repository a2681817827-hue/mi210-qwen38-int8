"""Give the CK tuner the exact INT8/FP32-scale/BF16 serving template."""
import argparse
from pathlib import Path


def patch(root):
    gen = root / "csrc/ck_gemm_a8w8/gen_instances.py"
    cu = root / "csrc/ck_gemm_a8w8/gemm_a8w8_tune.cu"
    old_gen = '[("B16", "B16"), ("F16", "F32"), ("F16", "F16")]'
    new_gen = '[("B16", "B16"), ("B16", "F32"), ("F16", "F32"), ("F16", "F16")]'
    old_cu = '''      TORCH_CHECK(x_scale.dtype() == at::ScalarType::BFloat16,
                  "I8 bf16-out tune path supports bf16 scales only, got: " +
                      std::string(c10::toString(x_scale.dtype())));
      // INT8 path, bf16 scales + bf16 out (upstream default)
      rowwise_dispatch<I8, B16, B16>(kernelId)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);'''
    new_cu = '''      if (x_scale.dtype() == at::ScalarType::Float)
      {
        rowwise_dispatch<I8, F32, B16>(kernelId)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
      }
      else
      {
        TORCH_CHECK(x_scale.dtype() == at::ScalarType::BFloat16,
                    "INT8 BF16 tuning requires FP32 or BF16 scales");
        rowwise_dispatch<I8, B16, B16>(kernelId)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
      }'''
    guard = '''  TORCH_CHECK(id < lookup.size(),
              "Kernel id " + std::to_string(id)  +" is out of range!");
  auto it = lookup.find(id);'''
    new_guard = '''  auto it = lookup.find(id);
  TORCH_CHECK(it != lookup.end(), "Requested CK tuning kernel was not built");'''
    tune_loop = '            for EDtype, DDtype in ' + new_gen + ':\n'
    pruned_loop = tune_loop + '''                if (self.dtype_filter is not None and
                    ("I8", DDtype, EDtype) not in self.dtype_filter):
                    continue
'''
    updates = {}
    for path, edits in ((gen, [(old_gen, new_gen), (tune_loop, pruned_loop)]),
                        (cu, [(old_cu, new_cu), (guard, new_guard)])):
        body = path.read_text()
        for old, new in edits:
            if new in body:
                continue
            if path == cu and old == old_cu and 'rowwise_dispatch<I8, F32, B16>' in body:
                continue
            if body.count(old) != 1:
                raise ValueError(f"Tuner source drift: {path.name}")
            body = body.replace(old, new)
        if path == cu:
            for scale, out in (('F32', 'B16'), ('B16', 'B16'), ('F32', 'F16'), ('F16', 'F16')):
                macro = f'AITER_BUILT_I8_{scale}_{out}'
                if f'#ifdef {macro}' in body:
                    continue
                call = f'rowwise_dispatch<I8, {scale}, {out}>(kernelId)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);'
                if body.count(call) != 1:
                    raise ValueError(f'Tuner dtype source drift: {scale}/{out}')
                body = body.replace(call, f'''#ifdef {macro}
      {call}
#else
      TORCH_CHECK(false, "CK tuning dtype is not supported by this pruned build");
#endif''')
        updates[path] = body
    for path, body in updates.items():
        backup = path.with_suffix(path.suffix + ".before-mi210-tuner")
        if not backup.exists():
            backup.write_text(path.read_text())
        path.write_text(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    patch(parser.parse_args().root)
