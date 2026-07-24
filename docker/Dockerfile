# ai-forensics — research container.
#
# Base image already ships Python 3.11 + torch 2.7.1 + torchvision built against
# CUDA 12.6.  Those are NEVER reinstalled here: a pip-resolved torch wheel would
# silently replace the CUDA build with a generic one and every GPU experiment in
# this testbed would start running on the CPU.  See the PIP_CONSTRAINT guard below.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# The project is bind-mounted, so the container user must match the HOST user that
# owns the checkout, or every generated artifact lands root-owned on the host.
# `make build` passes your live `id -u` / `id -g`; these are only fallbacks.
ARG UID=1000
ARG GID=1000
ARG USERNAME=aif
ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# --------------------------------------------------------------------------
# 1. OS packages (changes rarely → first layer)
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        htop \
        less \
        procps \
        sudo \
        tmux \
        tzdata \
        unzip \
        vim \
        wget \
    && git lfs install --system --skip-repo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# 2. Non-root user matching the host uid/gid
# --------------------------------------------------------------------------
RUN set -eux; \
    if ! getent group "${GID}" >/dev/null; then groupadd --gid "${GID}" "${USERNAME}"; fi; \
    if ! getent passwd "${UID}" >/dev/null; then \
        useradd --create-home --home-dir "/home/${USERNAME}" \
                --uid "${UID}" --gid "${GID}" --shell /bin/bash "${USERNAME}"; \
    fi; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/90-${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/90-${USERNAME}"; \
    mkdir -p "/home/${USERNAME}/.cache/huggingface" \
             "/home/${USERNAME}/.cache/torch/hub/checkpoints" \
             "/home/${USERNAME}/.local/bin" /workspace; \
    chown -R "${UID}:${GID}" "/home/${USERNAME}" /workspace

# Debian's /etc/profile hard-resets PATH, which drops /opt/conda/bin — so a login
# shell (`bash -l`, our default CMD) would lose `python`.  profile.d is sourced
# after that reset, so restoring the path here fixes login shells for good.
RUN printf '%s\n' 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/00-conda-path.sh \
    && chmod 0644 /etc/profile.d/00-conda-path.sh

# --------------------------------------------------------------------------
# 3. Freeze the preinstalled torch stack, then install our dependencies.
#    Only docker/requirements.txt is copied, so editing project source never
#    invalidates this (slow) layer — the source lives on a bind mount anyway.
# --------------------------------------------------------------------------
RUN python -c "import torch, torchvision; \
open('/etc/pip-torch-constraints.txt', 'w').write( \
'torch==%s\ntorchvision==%s\n' % (torch.__version__.split('+')[0], torchvision.__version__.split('+')[0]))" \
    && cat /etc/pip-torch-constraints.txt

# Applies to every pip run in this image, including interactive ones by the user:
# nothing may drag in a different torch/torchvision build.
ENV PIP_CONSTRAINT=/etc/pip-torch-constraints.txt

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Build-time gate: fail the build (not the first experiment) if the CUDA-enabled
# torch was clobbered by a transitive dependency.
RUN python -c "import torch, torchvision, transformers; \
assert torch.version.cuda, 'torch lost its CUDA build during pip install'; \
print('torch', torch.__version__, '| torchvision', torchvision.__version__, \
'| cuda', torch.version.cuda, '| transformers', transformers.__version__)"

# --------------------------------------------------------------------------
# 4. Runtime environment
# --------------------------------------------------------------------------
# TORCH_HOME is the second cache root and is easy to forget: torchvision weights
# (ResNet18/50) are NOT HuggingFace artifacts — `get_state_dict()` writes to
# $TORCH_HOME/hub/checkpoints, so without this (and the matching bind mount) they
# re-download on every fresh container and HF_HUB_OFFLINE=1 cannot cover them.
ENV PYTHONPATH=/workspace/src \
    HF_HOME=/home/${USERNAME}/.cache/huggingface \
    TORCH_HOME=/home/${USERNAME}/.cache/torch \
    AIF_PROJECT_ROOT=/workspace \
    TOKENIZERS_PARALLELISM=false

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

USER ${USERNAME}
WORKDIR /workspace

# No HEALTHCHECK on purpose: this is a long-lived interactive dev box whose main
# process is a shell.  There is no service to probe, restart policy is "no", so a
# health state would carry no signal — the entrypoint banner reports the facts
# that actually matter (torch/CUDA/GPU/HF-cache) once, at start.

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash", "-l"]
