# Requires the saved local image. All instructions below are metadata only.
ARG BASE_IMAGE=int8-vllm-mi210:saved-container-20261008
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/a2681817827-hue/mi210-qwen38-int8"
LABEL org.opencontainers.image.description="Saved MI210 TP1/C1 48K control: five-run median 102.56 tok/s"
CMD ["--help"]
