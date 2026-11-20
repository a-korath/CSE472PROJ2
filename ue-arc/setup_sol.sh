#!/bin/bash
# One-time Sol setup (idempotent: safe to re-run). Course rule: run it INSIDE a compute session
# (the VS Code Tunnel terminal, prompt [<ASURITE>@sgXXX ...]), never on a login node.
#   1. creates the `ue` env from environment.yml, stored on scratch (/scratch/$USER/envs/ue) so it doesn't fill home
#   2. downloads every model the jobs load (jobs run with HF_HUB_OFFLINE=1) + WordNet
#   3. checks torch sees the GPU
# Prereq in the same shell: export HF_HOME=/scratch/$USER/hf ; huggingface-cli login (after the env exists, see runbook).
set -eo pipefail
trap 'echo "SETUP FAILED at line $LINENO (exit $?). See the lines above."' ERR
export HF_HUB_DISABLE_PROGRESS_BARS=1   # progress bars overwrite lines and hide errors in logs
step() { echo; echo "=== [setup] $* ==="; }
case "$(hostname)" in login*) echo "ERROR: you are on a login node. Run this in the VS Code Tunnel (compute node) terminal."; exit 1;; esac
step "1/5 load mamba"
module load mamba/latest >/dev/null 2>&1 || module load mamba/latest
mkdir -p /scratch/$USER/envs
conda config --append envs_dirs /scratch/$USER/envs 2>/dev/null || true   # so `source activate ue` finds it by name
step "2/5 environment (/scratch/$USER/envs/ue)"
if [ ! -d /scratch/$USER/envs/ue ]; then mamba env create -p /scratch/$USER/envs/ue -f environment.yml; fi
export PATH=/scratch/$USER/envs/ue/bin:$PATH   # `source activate` is a no-op in non-interactive shells on Sol
echo "python = $(command -v python)"
python -c 'import huggingface_hub' || { echo 'ERROR: env not usable'; exit 1; }
set -u
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}
mkdir -p "$HF_HOME" logs results adapters
step "3/5 Hugging Face login (HF_HOME=$HF_HOME)"
if ! python -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 1)"; then
  echo "Env ready. Now log in to Hugging Face (token is saved under HF_HOME=$HF_HOME), then re-run this script:"
  echo "  export PATH=/scratch/\$USER/envs/ue/bin:\$PATH && hf auth login && bash setup_sol.sh"
  exit 2
fi
step "4/5 GPU check + model downloads (~35 GB, 20-40 min)"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)
from huggingface_hub import snapshot_download
for repo, skip in [("meta-llama/Meta-Llama-3-8B", ["original/*", "*.pth", "*.bin"]),   # original/ = 16 GB duplicate
                   ("Qwen/Qwen3-8B", ["*.bin"]),
                   ("openai-community/gpt2", ["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.bin"]),
                   ("sentence-transformers/all-MiniLM-L6-v2", ["onnx/*", "openvino/*"])]:
    print("downloading", repo, "...", flush=True)
    p = snapshot_download(repo, ignore_patterns=skip)
    print("   done:", p, flush=True)
import nltk; nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
PY
step "5/5 done"
echo "setup OK"
