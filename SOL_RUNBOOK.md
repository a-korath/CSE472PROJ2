# Sol runbook: get the training running

Who runs what:
- **Aryan:** baselines (zero-shot + clean fine-tunes, both models × 3 seeds, plus a 16-question protection smoke test).
- **Arjun:** protection (all 5 rounds), then victims (random + protected, both models × 3 seeds), chained automatically.

Each of you works in your **own** Sol account and scratch space. That doubles the GPUs you can run at once.

---

## Part 1. What the course allows us to request

From the CSE 472 Sol guide (`ASU_Sol_VSCode_Guideline.pdf`, Step 5):

| Setting | Course rule | What our jobs request |
|---|---|---|
| Account | `class_cse47267488fall2026` | same (in every script) |
| Partition | `htc` for work finishing within ~4 h; `public`/`general` for longer | baselines + victims → `htc`; protection → `general` |
| CPU | start with 8 cores | 8 |
| Memory | start with 64 GiB | 64 GiB |
| GPU | `gpu:a100:1` | 1 × A100 |
| Wall time | enough for the work; match it to the partition | baselines 4 h, victims 4 h, protection 12 h |

We stay at the course defaults, so no mentor approval is needed. Don't add `-C a100_80` (80 GB only). It lengthens the queue, and 40 GB is enough for an 8B model with LoRA.

Other rules that matter:
- **Nothing heavy on the login node.** Jobs go through `sbatch`. Setup (env + downloads) is allowed there but can be killed; Part 4 has the fallback.
- **Interactive web sessions idle for 4 h are cancelled.** We use batch jobs, so closing your laptop is fine.
- **No `sleep` loops in jobs.**

---

## Part 2. Connect to the ASU VPN (skip if you're on campus Wi-Fi)

1. Install **Cisco Secure Client (AnyConnect)**: search "ASU VPN" on asu.edu (ASU Service Center → VPN) and download the client for Windows.
2. Open Cisco Secure Client and type **`sslvpn.asu.edu`** in the address box → **Connect**.
3. Sign in with your **ASURITE** username and password, then approve the **Duo** push.
4. Wait until the client shows **Connected**. Leave it connected the whole time you're using Sol.

---

## Part 3. Open a terminal on Sol

**Option A, browser (easiest):**
1. Go to **https://ood05.sol.rc.asu.edu/pun/sys/dashboard/** (on VPN) and sign in with ASURITE + Duo.
2. Top menu → **Clusters** → **Sol Shell Access**. A terminal opens in a new tab. This is the **login node**.

**Option B, your own terminal:** `ssh <ASURITE>@login.sol.rc.asu.edu` (on VPN; same password + Duo).

Check you're on a login node: `hostname` shows something like `sol-login0x`.

---

## Part 4. One-time setup (both of you, ~30–45 min, mostly downloads)

**Arjun first:** GitHub → `a-korath/CSE472PROJ2` → Settings → Collaborators → **Add people** → `aryan285patyal`. Aryan accepts the email invite before cloning.

```bash
# 4.1 Get the code into scratch (big, fast, not your home quota)
cd /scratch/$USER
git clone https://github.com/a-korath/CSE472PROJ2.git
#   username = your GitHub username, password = a GitHub personal access token
#   (github.com → Settings → Developer settings → Personal access tokens → fine-grained,
#    read access to this repo). Already cloned? cd into it and `git pull`.
cd CSE472PROJ2/ue-arc
git log -1 --oneline        # must show 09a55a1 "Sol fixes..." or newer

# 4.2 Point the Hugging Face cache at scratch. Do this BEFORE logging in (the token is stored here)
export HF_HOME=/scratch/$USER/hf
echo 'export HF_HOME=/scratch/$USER/hf' >> ~/.bashrc     # so future shells have it too

# 4.3 Python environment
module avail mamba              # confirm a mamba module exists (the scripts assume mamba/latest)
module load mamba/latest
mamba create -n ue python=3.11 -y
source activate ue
pip install -r requirements.txt

# 4.4 Hugging Face login (your account must already have Llama-3 access approved)
huggingface-cli login           # paste a READ token from huggingface.co → Settings → Access Tokens

# 4.5 Download every model + WordNet (Llama-3-8B, Qwen3-8B, gpt2, all-MiniLM-L6-v2)
bash setup_sol.sh               # ends with "setup OK"; safe to re-run if interrupted
```

**If 4.5 gets killed on the login node**, run it as a small CPU job instead (it has no GPU, so it doesn't use GPU time):
```bash
mkdir -p logs
sbatch -A class_cse47267488fall2026 -p htc -c 4 --mem=16G -t 0-02:00 -o logs/setup-%j.out setup_sol.sh
squeue -u $USER                 # wait until it disappears, then:
tail logs/setup-*.out           # must end with "setup OK"
```
If that job fails with a network error, compute nodes have no internet. Go back to the login node and re-run `bash setup_sol.sh`; it resumes downloads.

---

## Part 5A. Aryan: submit baselines

```bash
cd /scratch/$USER/CSE472PROJ2/ue-arc
module load mamba/latest && source activate ue
mkdir -p logs                                   # must exist at submit time or SLURM drops the log

# Dry check: validates account/partition/GPU request without queuing
sbatch --test-only -p htc sol_baselines.sh
# Real submit (htc: course rule for jobs under ~4 h)
sbatch -p htc sol_baselines.sh
```
Expected: about 1–2 h of run time once it starts.

## Part 5B. Arjun: submit protection, then victims (chained)

```bash
cd /scratch/$USER/CSE472PROJ2/ue-arc
module load mamba/latest && source activate ue
mkdir -p logs

sbatch --test-only sol_protect.sh               # partition general, 12 h (longer than htc allows)
PJ=$(sbatch --parsable sol_protect.sh); echo "protect job = $PJ"

# Victims start automatically only if protection finishes successfully
sbatch -p htc --dependency=afterok:$PJ sol_victims.sh
```
Expected: protection ~6–10 h, then victims ~2–3 h.

**If `--test-only` complains about the QoS**, add `-q class` to that command and to the real submit (e.g. `sbatch -q class -p htc sol_baselines.sh`). If it complains about the partition for protection, try `-p public`.

---

## Part 6. Watching your jobs

| Command | What it shows |
|---|---|
| `squeue -u $USER` | Your jobs. `PD` = waiting in queue (REASON says why), `R` = running |
| `tail -f logs/arc-baselines-<jobid>.out` | Live log (Ctrl+C stops watching, not the job) |
| `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` | After it ends: `COMPLETED`, `FAILED`, `TIMEOUT`, `CANCELLED` |
| `scancel <jobid>` | Cancel a job |

You can close the terminal and your laptop; batch jobs keep running.

---

## Part 7. When something goes wrong

| Symptom | Fix |
|---|---|
| Log ends in `unbound variable` | Pull the latest code (`git pull`); the fix is in 09a55a1 |
| `OSError ... offline mode` / model not found | That model didn't download. Re-run `bash setup_sol.sh` on the login node |
| `401` / gated repo error for Llama | HF token missing or no Llama access. Redo 4.2 + 4.4 in the same shell, or use `LLAMA=NousResearch/Meta-Llama-3-8B sbatch ...` (same weights, ungated; say so in the report) |
| `CUDA out of memory` | You got a 40 GB card and it wasn't enough. Resubmit with `-C a100_80` (and tell your mentor you did) |
| Protection `TIMEOUT` | Its victims job is cancelled automatically. Re-run the two Part 5B submit lines; protection resumes from the last saved round |
| Baselines `TIMEOUT` on htc | Resubmit the same line; finished runs are skipped |

---

## Part 8. Friday morning checks

**Aryan:**
```bash
ls results/                                   # 2 *_zeroshot.json + 6 *_clean_s*.json
grep -h '"accuracy"' results/*zeroshot*.json results/*clean*.json
tail -20 logs/arc-baselines-*.out             # smoke test: protected loss < clean loss?
```
If Qwen3-8B's clean accuracy isn't at least ~3 points above its zero-shot, email the mentor about switching to `Qwen/Qwen3-8B-Base`.

**Arjun:**
```bash
cat protected/rounds.jsonl                    # every round: protected_loss well below clean_loss
ls protected/                                 # r0.jsonl, r1.jsonl, ... appear as rounds finish
squeue -u $USER                               # victims waiting (Dependency) or running
```
