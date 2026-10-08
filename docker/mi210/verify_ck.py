import sys
import torch

print("=== MI210 CK VERIFY ===")
print("torch:", torch.__version__)
print("HIP:", torch.version.hip)

assert torch.cuda.is_available(), "ROCm GPU unavailable"

name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)

print("GPU:", name)
print("device:", props)
assert props.gcnArchName.split(":", 1)[0] == "gfx90a", (
    f"Expected MI210/MI250 gfx90a, got {props.gcnArchName}"
)

try:
    import aiter
    print("AITER:", aiter.__file__)
except Exception as e:
    print("AITER IMPORT FAILED:", repr(e))
    sys.exit(10)

try:
    from aiter import gemm_a8w8_CK, pertoken_quant
    print("gemm_a8w8_CK: OK")
    print("pertoken_quant: OK")
except Exception as e:
    print("AITER CK IMPORT FAILED:", repr(e))
    sys.exit(11)

# Small real gfx90a execution test.
M, N, K = 16, 4096, 4096

x = torch.randn(
    (M, K),
    device="cuda",
    dtype=torch.float16,
)

w = torch.randint(
    -127, 128,
    (N, K),
    device="cuda",
    dtype=torch.int8,
)

try:
    xq, xs = pertoken_quant(
        x,
        quant_dtype=torch.int8,
    )

    ws = torch.ones(
        (1, N),
        device="cuda",
        dtype=torch.float32,
    )

    torch.cuda.synchronize()

    y = gemm_a8w8_CK(
        xq,
        w,
        xs,
        ws,
        None,
        torch.float16,
    )

    torch.cuda.synchronize()

    print("CK RUN: OK")
    print("shape:", tuple(y.shape))
    print("dtype:", y.dtype)
    print("finite:", bool(torch.isfinite(y).all()))
    assert y.shape == (M, N), f"Unexpected CK output shape: {y.shape}"
    assert y.dtype == torch.float16, f"Unexpected CK output dtype: {y.dtype}"
    assert bool(torch.isfinite(y).all()), "Non-finite CK output"

    # Check the GEMM itself using the quantized operands, so activation
    # quantization error does not hide a kernel or scale-layout defect.
    reference = (
        xq.float() @ w.float().T
    ) * xs.float().reshape(M, 1) * ws.float().reshape(1, N)
    torch.testing.assert_close(y, reference.to(y.dtype), rtol=2e-3, atol=2e-2)
    print("CK numerical reference: PASS")

except Exception as e:
    print("CK RUN FAILED:", repr(e))
    sys.exit(12)

print("=== CK GFX90A PASS ===")
