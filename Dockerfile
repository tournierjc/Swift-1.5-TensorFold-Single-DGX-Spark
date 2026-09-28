# TensorFold on one DGX Spark: the branch that reads the Swift 1.5 NVFP4 checkpoint as it ships.
#
#   scripts/build.sh                    build this image (local only, no registry)
#   scripts/pull.sh                     download the checkpoint into the host's Hugging Face cache
#   scripts/serve.sh                    serve it, foreground, OpenAI-compatible endpoint on :8080
#
# The image carries TensorFold and its dependencies only. Weights, the Hugging Face cache and the kernel
# caches are host bind mounts, so rebuilding the image never re-downloads the 186 GB checkpoint.

FROM nvcr.io/nvidia/pytorch:26.07-py3

# The revision under test. Pinned to the commit this repository was prepared against; override to test
# another one:  TF_REF=<sha|branch> scripts/build.sh  (or --build-arg TF_REF=...)
ARG TF_REPO=https://github.com/tournierjc/TensorFold.git
ARG TF_REF=c17df8a6464d9d4c856941bf858fb9b310b7636e

LABEL org.opencontainers.image.title="Swift 1.5 on TensorFold (single DGX Spark)" \
      org.opencontainers.image.source="https://github.com/tournierjc/Swift-1.5-TensorFold-Single-DGX-Spark" \
      org.opencontainers.image.description="TensorFold CUDA serving stack for the Swift 1.5 NVFP4 checkpoint" \
      org.opencontainers.image.licenses="MIT" \
      tf.repo="${TF_REPO}" tf.ref="${TF_REF}"

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/hf \
    TRITON_CACHE_DIR=/state/triton \
    TORCHINDUCTOR_CACHE_DIR=/state/inductor \
    TORCH_EXTENSIONS_DIR=/state/torch-extensions

# NVIDIA's container supplies CUDA, torch, triton and the extension compiler; the package has no `cuda`
# extra, and installing it here must not replace that toolchain.
RUN python3 -m pip install --no-cache-dir "tensorfold @ git+${TF_REPO}@${TF_REF}" \
    && python3 -c "import torch; print('torch', torch.__version__)" \
    && python3 -c "import tensorfold; print('tensorfold', tensorfold.__version__)" \
    && python3 -c "from tensorfold.families.qwen4_exp.cuda import nvfp4, nvfp4_moe; print('NVFP4 route present')" \
    && mkdir -p /hf /state /models

# /hf    Hugging Face cache (186 GB for this checkpoint)      -> host bind
# /state kernel caches: triton, torch extensions, inductor    -> host bind
# /models local checkpoints, read-only                        -> host bind
WORKDIR /models
EXPOSE 8080
ENTRYPOINT ["tensorfold"]
CMD ["--help"]
