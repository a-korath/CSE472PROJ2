#!/bin/bash
# Sol booking 2: min-min protection with the Llama-3-8B surrogate.
# Resumable: protect.py restarts from the last saved round (r*.jsonl, ckpt_r*/), so just resubmit if the slot is cut.
# Same Sol-specific lines as sol_baselines.sh; check them there.
#SBATCH -J arc-protect
#SBATCH -p general
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# Sol: module env + scratch cache. Anywhere else (RunPod, Colab): `HF_HOME=... bash -e` in an env with requirements.txt
if command -v module >/dev/null; then module load mamba/latest; source activate ue; fi   # check: module avail mamba
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}   # weights pre-downloaded here from the login node
LLAMA=${LLAMA:-meta-llama/Meta-Llama-3-8B}

# protected/neighbours.json is built on the laptop (protect.py --neighbours_only) and copied up; built here if missing
python protect.py --model "$LLAMA" --train data/train.jsonl --out protected
