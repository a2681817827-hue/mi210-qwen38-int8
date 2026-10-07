# Measured MI210 patch snapshots

`int8-vllm-mi210-measured.patch` reconstructs the selected framework and operator
sources from the repeated ~103 tok/s image relative to curvedinf/int8-vllm
commit `5226dbdbd055b645e0a2e972ddb54e97b839858b`.

It covers gfx90a dispatch, CK INT8 interfaces, the INT8 LM head, FP32-state
native GDN loader/hooks, G128 target/draft attention and the opt-in query tile.
The generated GDN source and its build provenance, CK selection report, AITER
wrapper and original licenses are archived under `operators/`.

Apply only to the identified base (`git apply --check` before `git apply`).
This is a source snapshot of a tested local image, not a claim of compatibility
with arbitrary newer vLLM, or a complete from-scratch build recipe. The pinned
Docker image contains the corresponding compiled GDN library and CK modules.
The local launch scripts continue to reuse that tested image and cache.
