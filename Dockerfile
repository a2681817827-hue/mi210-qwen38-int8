# This overlay requires the already measured local image; it is not a registry release.
ARG BASE_IMAGE=int8-vllm-mi210:query5-candidates-20261007
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="MI210 Qwen3.8 INT8 measured optimization"
LABEL org.opencontainers.image.description="INT8 W8A8 target + DFlash2, G128 KV, FP32 Mamba; performance candidate"
ENV VLLM_MI210_G128_QUERY_TILE=5 VLLM_G128_DECODE_CDNA2_TILE=64 VLLM_MI210_FLASH_SPLITS=32
ENV VLLM_MI210_INT8_LM_HEAD=1 VLLM_MI210_NATIVE_GDN=1 VLLM_MI210_INT8_QK=1 AITER_MI210_TUNED_CK=1
LABEL org.opencontainers.image.source="https://github.com/a2681817827-hue/mi210-qwen38-int8"
CMD ["--help"]
