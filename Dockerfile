# TensorFold on one DGX Spark: the branch that reads the Swift 1.5 NVFP4 checkpoint as it ships.
#
#   scripts/build.sh                    build this image (local only, no registry)
#   scripts/pull.sh                     download the checkpoint into the host's Hugging Face cache
#   scripts/serve.sh                    serve it, foreground, OpenAI-compatible endpoint on :8083
#
# The image carries TensorFold and its dependencies only. Weights, the Hugging Face cache and the kernel
# caches are host bind mounts, so rebuilding the image never re-downloads the 186 GB checkpoint.

FROM nvcr.io/nvidia/pytorch:26.07-py3

# The revision under test. Pinned to the commit this repository was prepared against -- upstream 0.6.0 plus the
# rig's own changes, which live in `integration/0.6.0` of the fork `tournierjc/TensorFold` (the "Engine revision
# and the branches" section of the README lists what each one carries). Override to test another one:
#   TF_REF=<sha|branch> scripts/build.sh  (or --build-arg TF_REF=...)
ARG TF_REPO=https://github.com/tournierjc/TensorFold.git
ARG TF_REF=c3fa14f4cdd2454d32326c6bc2845a71cb76e7b6

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
# --vision needs two more: the tower's modules live in transformers (the same tower the dense
# Qwen3.5/3.8 checkpoints use), and PyAV decodes video. transformers is pinned to the version the
# patch was written against, and the import is the check - a missing transformers only shows up when
# the first image arrives, nine minutes into a load.
RUN python3 -m pip install --no-cache-dir "tensorfold @ git+${TF_REPO}@${TF_REF}" \
    && python3 -m pip install --no-cache-dir "transformers==5.17.0" av \
    && python3 -c "import torch; print('torch', torch.__version__)" \
    && python3 -c "import tensorfold; print('tensorfold', tensorfold.__version__)" \
    && python3 -c "from tensorfold.families.qwen4_exp.cuda import nvfp4, nvfp4_moe; print('NVFP4 route present')" \
    && python3 -c "from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel; print('vision tower modules present')" \
    && mkdir -p /hf /state /models

# The engine revision carries every change this rig serves (the README's "Engine revision and the branches"
# section names them). The build asserts the one that is silent when missing: the MTP draft head's reduced
# vocabulary, `families/qwen4_exp/cuda/draft_vocab.txt`, 80,014 ids -- the pinned revision's own list, which is
# what `scripts/build_draft_vocab.py` produced. A build against a ref that does not carry the port (upstream, or
# an older tag) fails here in seconds instead of scoring 79,591 ids at run time -- the failure mode that cost the
# sibling recipe ~17%, and the reason this check is a build step and not a flag.
RUN pkg="$(python3 -c 'import tensorfold, os; print(os.path.dirname(tensorfold.__file__))')" \
    && PKG="${pkg}" python3 -c "import os, pathlib; p = pathlib.Path(os.environ['PKG'], 'families/qwen4_exp/cuda/draft_vocab.txt'); ids = [int(x) for x in p.read_text().split()]; assert ids and ids == sorted(set(ids)), p; assert len(ids) == 80014, f'{p}: {len(ids)} ids, the port carries 80014'; print('[build] draft vocabulary:', len(ids), 'ids')"

# /hf    Hugging Face cache (186 GB for this checkpoint)      -> host bind
# /state kernel caches: triton, torch extensions, inductor    -> host bind
# /models local checkpoints, read-only                        -> host bind
WORKDIR /models
EXPOSE 8083
ENTRYPOINT ["tensorfold"]
CMD ["--help"]
