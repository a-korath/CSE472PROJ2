# Sol runbook: from laptop to training jobs

Based on the CSE 472 Sol guide (`ASU_Sol_VSCode_Guideline.pdf`) and a tested connection on 2026-11-17
(Windows + VS Code → compute node `sg024`, A100-SXM4-80GB).

**The idea:** you open **one short VS Code Tunnel session** (a compute node with a GPU) to set up and submit.
The actual training runs as **batch jobs** that keep going after you close everything. Then you end the session.

| Who | Submits | Runs for |
|---|---|---|
| **Aryan** | baselines (zero-shot + clean fine-tunes, both models × 3 seeds, + 16-question protection smoke test) | ~1–2 h |
| **Arjun** | protection (5 rounds) → victims (random + protected, both models × 3 seeds), chained | ~6–10 h, then ~2–3 h |

Each of you uses your **own ASURITE, own GitHub account, own Hugging Face account**. Two accounts = two GPU allowances.

---

## 0. Before you start (5 min, both)

- [ ] **Arjun:** GitHub → `a-korath/CSE472PROJ2` → Settings → Collaborators → **Add people** → `aryan285patyal`. **Aryan:** accept the email invite.
- [ ] **Both:** Hugging Face account with **Llama-3 access approved** (huggingface.co/meta-llama/Meta-Llama-3-8B shows "You have been granted access"). Create a **Read** token: huggingface.co → Settings → Access Tokens → New token → Read. Keep it handy; you paste it once.
- [ ] **Both:** GitHub **personal access token** for cloning on Sol: github.com → Settings → Developer settings → Fine-grained tokens → Generate → Repository access: `a-korath/CSE472PROJ2` → Contents: Read-only.
- [ ] VS Code installed with these 4 extensions (PowerShell):
  ```powershell
  code --install-extension ms-vscode-remote.remote-ssh
  code --install-extension ms-vscode-remote.remote-ssh-edit
  code --install-extension ms-vscode.remote-explorer
  code --install-extension ms-vscode.remote-server
  code --list-extensions | findstr remote      # all 4 listed
  ```
  The course guide lists only the first 3; Remote - Tunnels (`ms-vscode.remote-server`) is also required.

---

## 1. VPN (skip on campus Wi-Fi)

1. Install **Cisco Secure Client** from https://sslvpn.asu.edu (or ASU MyApps).
2. Open it → server **`sslvpn.asu.edu/2fa`** → **Connect** → ASURITE + password → approve **Duo**.
3. Test: open https://ood05.sol.rc.asu.edu/pun/sys/dashboard/ → the OnDemand dashboard loads.
   (`ping sol.asu.edu` timing out is normal; ping is blocked. The dashboard is the real test.)
4. Keep the VPN connected while starting and using the session.

---

## 2. Launch the setup session (browser, not VS Code)

Dashboard → gold top bar → **Interactive Apps** → under **Tunnels** → **VSCode Tunnel** (*not* "VSCode Server").

Fill the form exactly like this (course defaults; no mentor approval needed):

| Field | Value | Why |
|---|---|---|
| Account | `class_cse47267488fall2026` | course allocation |
| Partition | `htc` | course rule: work under ~4 h goes on htc (Sol may reroute to `public`; that's fine) |
| QoS | `class` (default) | the class account's QoS |
| CPU Core Allocation | `8` | course default |
| Memory Allocation (GiB) | `64` | course default |
| GPU Resources | `gpu:a100:1` | full A100; needed so mamba installs the CUDA build of PyTorch and we can test it |
| Wall time | `0-3` (= 3 hours, days-hours) | env install + ~35 GB of downloads take 45–90 min; don't start installs with little time left |
| Additional sbatch options | *(blank)* | |
| VSCode Instance | `Primary` | |
| ☑ Email when the session starts | ticked | queue can take a while |

Click **Launch**.

## 3. Connect VS Code to it

1. Browser → **My Interactive Sessions** (https://ood05.sol.rc.asu.edu/pun/sys/dashboard/batch_connect/sessions). Wait for the card to say **Running** with Host `sgXXX.sol.rc.asu.edu`. No code shown yet? Wait ~30 s and refresh.
2. Click **Copy Code and Login via GitHub** → sign in to **your** GitHub → paste the code → Continue → **Authorize Visual Studio Code**. Never share this code.
3. Desktop VS Code → **Remote Explorer** → dropdown **Remotes (Tunnels/SSH)** → sign in with **GitHub** (same account) → Refresh → **Tunnels** → `sol_<ASURITE>` (running) → click →.
4. Open the terminal: **Ctrl + `**. The prompt looks like `[<ASURITE>@sgXXX ~]$`.

Sanity check (paste):
```bash
hostname          # sgXXX  (a compute node; NOT login*)
nvidia-smi        # NVIDIA A100 (80 GB or 40 GB)
```
If the dashboard shows a red **"Slurm Errors"** banner, that's a Research Computing outage. Wait and retry; don't change settings.

---

## 4. One-time setup inside the tunnel terminal (both, ~45–90 min)

```bash
# 4.1 Hugging Face cache on scratch (weights ~35 GB; scratch, not home).
#     If an earlier test put HF_HOME=.../hf_cache in ~/.bashrc, this replaces it so everything uses one path.
sed -i '/HF_HOME/d' ~/.bashrc
echo 'export HF_HOME=/scratch/$USER/hf' >> ~/.bashrc
source ~/.bashrc && echo $HF_HOME          # /scratch/<ASURITE>/hf

# 4.2 Code on scratch
cd /scratch/$USER
git clone https://github.com/a-korath/CSE472PROJ2.git
#   Username: your GitHub username.  Password: the fine-grained token from step 0.
#   Already cloned?  cd CSE472PROJ2 && git pull
cd CSE472PROJ2/ue-arc
git log -1 --oneline                       # must be the "Sol: QoS class, mamba env..." commit or newer

# 4.3 Create the environment (mamba, from environment.yml; stored in /scratch/$USER/envs/ue).
#     First run stops after creating the env and asks you to log in to Hugging Face. That's expected.
bash setup_sol.sh

# 4.4 Hugging Face login (paste the Read token; answer "n" to "add as git credential")
source activate ue
huggingface-cli login

# 4.5 Run setup again: checks the GPU and downloads Llama-3-8B, Qwen3-8B, gpt2, all-MiniLM-L6-v2, WordNet
bash setup_sol.sh
#   expect early:  torch 2.x  cuda: True  NVIDIA A100...
#   expect last:   setup OK
```
If the session runs out of time mid-download, start a new session (step 2) and re-run `bash setup_sol.sh`; it resumes. If it ran out mid-**env install**, the env may be broken: `rm -rf /scratch/$USER/envs/ue` and run 4.3 again with plenty of wall time.

---

## 5. Submit the training jobs (from the same tunnel terminal)

The job scripts already carry the right request in their `#SBATCH` headers:

| Script | Account | Partition / QoS | CPU / Mem / GPU | Wall time |
|---|---|---|---|---|
| `sol_baselines.sh` | class_cse47267488fall2026 | **htc** (passed at submit) / class | 8 / 64G / 1×A100 | 4 h |
| `sol_protect.sh` | class_cse47267488fall2026 | general / class | 8 / 64G / 1×A100 | 12 h (longer than htc allows, so general per the course rule) |
| `sol_victims.sh` | class_cse47267488fall2026 | **htc** (passed at submit) / class | 8 / 64G / 1×A100 | 4 h (passed at submit) |

### 5A. Aryan
```bash
cd /scratch/$USER/CSE472PROJ2/ue-arc
mkdir -p logs                                   # SLURM drops the log if this folder is missing at submit time
sbatch --test-only -p htc sol_baselines.sh      # validates the request; no job is queued
sbatch -p htc sol_baselines.sh                  # prints "Submitted batch job <ID>"
squeue -u $USER                                 # PD = waiting, R = running
```

### 5B. Arjun
```bash
cd /scratch/$USER/CSE472PROJ2/ue-arc
mkdir -p logs
sbatch --test-only sol_protect.sh
PJ=$(sbatch --parsable sol_protect.sh); echo "protect job = $PJ"
sbatch -p htc -t 0-04:00 --dependency=afterok:$PJ sol_victims.sh   # starts only if protection succeeds
squeue -u $USER                                 # victims show PD (Dependency) until protection ends
```

If `--test-only` rejects anything: htc time limit → keep `-t 0-04:00`; partition `general` refused → add `-p public`.
If `sbatch: command not found` in the tunnel, submit from Dashboard → **Clusters → Sol Shell Access** instead (submitting from a login node is allowed; installing is not).

---

## 6. End the setup session (releases the GPU)

The batch jobs are independent; they keep running without the tunnel.
1. VS Code: bottom-left → **Close Remote Connection**.
2. Browser → **My Interactive Sessions** → the VSCode Tunnel card → red **Delete** → confirm the card is gone.

Portal sessions also auto-cancel after 4 h idle, but end it yourself so the GPU goes back to the class.

---

## 7. Checking on jobs later (no GPU needed)

Use Dashboard → **Clusters → Sol Shell Access** (a login node; fine for monitoring), then `cd /scratch/$USER/CSE472PROJ2/ue-arc`.

| Command | Shows |
|---|---|
| `squeue -u $USER` | your jobs: `PD` waiting (REASON says why), `R` running |
| `tail -f logs/arc-baselines-<ID>.out` | live log (Ctrl+C stops watching, not the job) |
| `sacct -j <ID> --format=JobID,State,Elapsed,MaxRSS` | after it ends: COMPLETED / FAILED / TIMEOUT / CANCELLED |
| `scancel <ID>` | cancel a job |

Want an email when a job ends or fails? Add `--mail-type=END,FAIL` to the `sbatch` line (it goes to your ASU address).

---

## 8. When something goes wrong

| Symptom in the log | Fix |
|---|---|
| `unbound variable` | `git pull` (fixed in the repo) and resubmit |
| `OSError ... offline mode` / model not found | That model isn't in `$HF_HOME`. New tunnel session → `bash setup_sol.sh` |
| `401` / gated repo for Llama | token or Llama access missing. Redo 4.4; or prefix the submit with `LLAMA=NousResearch/Meta-Llama-3-8B` (same weights, ungated; disclose in the report) |
| `CUDA out of memory` | you got a 40 GB slice. Resubmit with `-C a100_80` added and tell the mentor |
| protection `TIMEOUT` | the chained victims job cancels itself. Re-run the two 5B submit lines; protection resumes from its last saved round |
| baselines/victims `TIMEOUT` on htc | resubmit the same line; finished runs are skipped |
| `ModuleNotFoundError` | env incomplete. New tunnel session → `bash setup_sol.sh` |

---

## 9. Friday morning checks

**Aryan** (Sol Shell):
```bash
cd /scratch/$USER/CSE472PROJ2/ue-arc
ls results/                                    # 2 *_zeroshot.json + 6 *_clean_s*.json
grep -H '"accuracy"' results/*zeroshot*.json results/*clean*.json
grep -iE "loss|final" logs/arc-baselines-*.out | tail     # smoke test: protected loss < clean loss
```
If Qwen3-8B's clean accuracy isn't ~3+ points above its zero-shot, email the mentor about `Qwen/Qwen3-8B-Base`.

**Arjun:**
```bash
cd /scratch/$USER/CSE472PROJ2/ue-arc
cat protected/rounds.jsonl      # every round: protected_loss well below clean_loss
ls protected/                   # r0.jsonl, r1.jsonl, ... as rounds finish
squeue -u $USER                 # victims: PD (Dependency) or R
```

---

## Course rules this follows
VPN off campus · VSCode **Tunnel** (not Server) · account `class_cse47267488fall2026` · htc for < 4 h, general for longer ·
8 cores / 64 GiB / `gpu:a100:1` · envs via **mamba**, created **inside a compute session**, recorded in `environment.yml` ·
scratch for large files (not backed up; unused files deleted after 90 days, fine for model weights) · cancel sessions you're done with.
