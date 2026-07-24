#!/usr/bin/env bash
# ai-forensics container entrypoint.
#
# Two jobs, both cheap:
#   1. make sure the managed directory skeleton exists on the bind mount
#   2. print the handful of facts that decide whether an experiment is valid at
#      all — torch/CUDA build, which GPUs are visible, and whether the 30 GB host
#      HuggingFace cache actually got mounted (if it did not, every download runs
#      again into a throwaway layer).
set -euo pipefail

PROJECT_ROOT="${AIF_PROJECT_ROOT:-/workspace}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"

hr() { printf '%s\n' "------------------------------------------------------------"; }

# --- 1. directory skeleton -------------------------------------------------
# Preferred path: reuse aiforensics.paths.ensure_dirs so there is exactly one
# definition of the layout. Falls back to mkdir when the source is not mounted.
ensure_dirs() {
    if python3 -c "from aiforensics.paths import ensure_dirs; ensure_dirs()" 2>/dev/null; then
        return 0
    fi
    mkdir -p \
        "${PROJECT_ROOT}"/data/{raw,processed,poisoned,rag} \
        "${PROJECT_ROOT}"/models/{pretrained,finetuned,adapters,tampered} \
        "${PROJECT_ROOT}"/outputs/{runs,logs,reports} 2>/dev/null || true
}

# --- 2. banner -------------------------------------------------------------
torch_report() {
    python3 - <<'PY' 2>&1 || true
import os

try:
    import torch
except Exception as exc:  # pragma: no cover - container smoke path
    print(f"  torch          : IMPORT FAILED ({exc})")
    raise SystemExit(0)

print(f"  torch          : {torch.__version__}  (cuda {torch.version.cuda})")
available = torch.cuda.is_available()
print(f"  cuda available : {available}")
print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

count = torch.cuda.device_count() if available else 0
for i in range(count):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu[{i}]         : {p.name}  {p.total_memory / 1024**3:.0f} GiB")
if not available:
    print("  gpu            : none visible — running CPU-only")

if count > 1:
    print("")
    print(f"  ###  WARNING: {count} GPUs are visible to this container.")
    print("  ###  This project is scoped to a SINGLE GPU (see AIF_GPU_ID in .env).")
    print("  ###  On a shared machine the extra device may belong to someone else.")
    print("  ###  Stop, fix CUDA_VISIBLE_DEVICES / the compose")
    print("  ###  deploy.resources.reservations.devices entry, and restart.")
PY
}

# A bind-mount source that Docker had to create itself is owned by root, and the
# aif user's writes then fail deep inside a download. Surface it at start instead.
writable_or_warn() {
    local dir="$1"
    [[ -w "${dir}" ]] && return 0
    echo "  ###  WARNING: ${dir} is not writable by $(id -un)"
    echo "  ###  (owner $(stat -c '%U:%G' "${dir}" 2>/dev/null || echo unknown))."
    echo "  ###  Docker creates a missing bind-mount source as root. On the HOST:"
    echo "  ###    mkdir -p <host dir> && sudo chown -R $(id -u):$(id -g) <host dir>"
    echo "  ###  then 'make down && make up'."
    return 0
}

# Authoritative "is this path a bind mount?" test — the image pre-creates both
# cache directories, so their mere existence proves nothing.
is_mounted() {
    grep -q " $1 " /proc/self/mountinfo 2>/dev/null
}

hf_cache_report() {
    local hub="${HF_HOME}/hub"
    if ! is_mounted "${HF_HOME}"; then
        echo "  hf cache       : ${HF_HOME} (NOT MOUNTED)"
        echo "  ###  WARNING: the host HuggingFace cache is not mounted here."
        echo "  ###  Everything will be re-downloaded into a throwaway layer."
        echo "  ###  Check the volume mapping in docker/docker-compose.yml."
        return
    fi
    local repos
    repos="$(find "${hub}" -mindepth 1 -maxdepth 1 -type d \
        \( -name 'models--*' -o -name 'datasets--*' \) 2>/dev/null | wc -l)"
    echo "  hf cache       : ${HF_HOME} (${repos} cached repos)"
    if [[ -n "${HF_TOKEN:-}" ]]; then
        echo "  hf token       : present (env)"
    elif [[ -f "${HF_HOME}/token" ]]; then
        echo "  hf token       : present (cache file)"
    else
        echo "  hf token       : absent — gated repos will 401"
    fi
    writable_or_warn "${HF_HOME}"
}

# torchvision weights (ResNet18/50) never touch the HF cache: torch.hub writes
# them to $TORCH_HOME/hub/checkpoints. Reported separately so a missing mount is
# visible before it costs a re-download / breaks an offline run.
torch_cache_report() {
    local ckpt="${TORCH_HOME}/hub/checkpoints"
    mkdir -p "${ckpt}" 2>/dev/null || true
    if ! is_mounted "${TORCH_HOME}"; then
        echo "  torch cache    : ${TORCH_HOME} (NOT MOUNTED)"
        echo "  ###  WARNING: torchvision checkpoints are not persisted here."
        echo "  ###  ResNet weights will re-download and HF_HUB_OFFLINE=1 will not"
        echo "  ###  cover them. Check the volume mapping in docker-compose.yml."
        return
    fi
    local ckpts
    ckpts="$(find "${ckpt}" -mindepth 1 -maxdepth 1 -type f \
        \( -name '*.pth' -o -name '*.pt' \) 2>/dev/null | wc -l)"
    echo "  torch cache    : ${TORCH_HOME} (${ckpts} checkpoints)"
    writable_or_warn "${TORCH_HOME}"
}

ensure_dirs

hr
echo "  ai-forensics dev container"
echo "  user           : $(id -un) (uid=$(id -u) gid=$(id -g))"
echo "  workdir        : $(pwd)   project=${PROJECT_ROOT}"
if [[ -d "${PROJECT_ROOT}/src/aiforensics" ]]; then
    echo "  package        : importable via PYTHONPATH=${PYTHONPATH:-<unset>}"
else
    echo "  package        : ###  WARNING: ${PROJECT_ROOT}/src/aiforensics missing —"
    echo "                   the project does not look bind-mounted."
fi
torch_report
hf_cache_report
torch_cache_report
hr

exec "$@"
