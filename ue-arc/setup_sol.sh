#!/bin/bash
# One-time Sol setup (idempotent: safe to re-run). Run from ue-arc/ on the LOGIN node after `huggingface-cli login`.
# Creates the `ue` env, downloads every model the jobs load (compute nodes run with HF_HUB_OFFLINE=1), and WordNet.
# If the login node kills the downloads, run it as a short job instead:
#   sbatch -A class_cse47267488fall2026 -p htc -c 4 --mem=16G -t 0-02:00 -o logs/setup-%j.out setup_sol.sh
set -eo pipefail
module load mamba/latest
if ! conda env list | grep -qE '^ue\s'; then mamba create -n ue python=3.11 -y; fi
source activate ue
set -u
pip install -q -r requirements.txt
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}
mkdir -p "$HF_HOME" logs results adapters
python - <<'PY'
from huggingface_hub import snapshot_download
skip = ["original/*", "*.pth", "*.bin", "*.gguf"]   # safetensors only; Llama's original/ is a 16 GB duplicate
for repo in ["meta-llama/Meta-Llama-3-8B", "Qwen/Qwen3-8B", "gpt2", "sentence-transformers/all-MiniLM-L6-v2"]:
    print("downloading", repo, flush=True)
    snapshot_download(repo, ignore_patterns=skip if repo != "gpt2" else ["*.h5", "*.msgpack", "*.onnx", "*.tflite"])
import nltk; nltk.download("wordnet"); nltk.download("omw-1.4")
print("setup OK")
PY
