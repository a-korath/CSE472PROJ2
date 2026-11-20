# CSE 472 Project: Unlearnable ARC-Challenge, progress update

**As of:** 2026-11-19 · **Deadline:** 2026-11-23 · **Guideline:** `Project_Guideline_1.pdf` (repo root) · **Code:** `ue-arc/`

## TL;DR
- All code is written and has been smoke-tested on a small model (Qwen3-0.6B, 4 examples). It has **not yet run on the real 8B models**.
- The laptop-side work is done: the ARC data splits, the word-neighbour lists, and the **random-edit control set** (all 1119 training examples).
- We have Llama-3-8B access on Sol, and Sol time is not a constraint. **The next step is to submit the three Sol jobs** (baselines, then protection, then victims).
- No GPU results yet, so the results table is still empty.

## 1. What we are building (locked design)
Huang et al.'s min–min ("error-minimizing") unlearnable examples, adapted to LLM fine-tuning:

$$\min_{\delta}\ \min_{\theta}\ \sum_i \ell_{\text{ans}}\big(f_\theta(x_i \oplus \delta_i),\ y_i\big)$$

subject to at most $k=2$ word swaps per example, $\text{SBERT}(x, x\oplus\delta) \ge 0.9$ and $\text{PPL}_{\text{GPT-2}}(x\oplus\delta) \le 1.2\times \text{PPL}_{\text{GPT-2}}(x)$.

Here $x$ = question + answer choices, and $y$ = the answer letter (never changed).

In plain words: we edit 2 words per question so the model already finds the right answer easy. When the loss is already near zero, fine-tuning has almost no gradient to learn from, so the attacker's model gains little from our data.

| Component | Our choice |
|---|---|
| **Loss** | Next-token cross-entropy on the **answer-letter token only** |
| **Surrogate** | Llama-3-8B base + LoRA, with exactly the victim settings (r=16, α=32, dropout 0.05, all q/k/v/o/gate/up/down projections, lr 1e-4, batch 8, bf16) |
| **Perturbation** | Whole-word swaps anywhere in the question or choices. Replacements come from the original word's closest neighbours in Llama-3's embedding space. |
| **Search** | HotFlip-style: rank swaps by the first-order loss change $(e_{\text{new}} - e_{\text{old}})^\top \nabla_e \ell$, keep the top 20, drop those failing the SBERT/PPL filters, re-score the rest exactly, keep the best swap only if it lowers the loss. Done greedily, twice. |
| **Loop** | Round 0 searches edits against the base model. Each later round trains the surrogate 2 epochs on the current protected set, then searches again. Up to 5 rounds, or stop once the surrogate's train accuracy is ≥ 99%. Every round is saved. |
| **Control** | Random edits: same k=2 budget, same neighbour lists, same filters, but random positions and random replacements |
| **Evaluation** | Test accuracy on clean ARC test (1172 questions) for zero-shot, Original, Random and Protected. Llama-3-8B is white-box, Qwen3-8B is black-box. 3 seeds, mean ± std. GPT-2 perplexity and SBERT (`all-MiniLM-L6-v2`) for readability and meaning. |
| **Success** | The protected fine-tune gains little or nothing over zero-shot, while the Original fine-tune does gain |

## 2. Done so far
- **Baseline code** (`data.py`, `eval.py`, `train_lora.py`, `sol_baselines.sh`): the dry run on Qwen3-0.6B passed. Zero-shot scored 0.69 and the clean LoRA run 0.63 on 16 examples; the numbers are meaningless at this size and only prove the plumbing works.
- **New code** (all smoke-tested end to end on Qwen3-0.6B: search → train → search, plus resuming after a cut):
  - `protect.py`: the min–min loop. It resumes from the last saved round. `--neighbours_only` builds the neighbour lists on a CPU.
  - `random_edits.py`: the random-edit control.
  - `metrics.py`: GPT-2 perplexity, SBERT similarity, and **where the edits land** (question / correct option / wrong option).
  - `summarize.py`: the mean ± std results table.
  - `sol_protect.sh`, `sol_victims.sh`: Sol jobs. They resume when resubmitted, and also run as plain `bash` off Sol.
- **Bugs fixed:**
  - SLURM needs `logs/` to exist *at submit time*, or the job's output is silently lost.
  - On a no-GPU laptop, `device_map="auto"` offloaded part of the model to disk and broke backprop, so the code now loads on CPU when there's no CUDA.
  - Wrong exit code in `--neighbours_only`.
- **Data ready for Sol:**
  - `data/{train,validation,test}.jsonl` (1119 / 299 / 1172 examples).
  - `protected/neighbours.json`: 50 neighbours for each of 4,350 words, built from Llama-3's embeddings.
  - `data/random_k2.jsonl`: the random control.

### Smoke-test signal (Qwen3-0.6B, 4 examples, not a result)
Round 0 edits cut the surrogate's answer loss from **0.94 → 0.24**. The optimization is doing what it should.

### Random-edit control (all 1119 training examples)
| Metric | Value |
|---|---|
| Edits per example | 1.87 (budget 2) |
| SBERT similarity | 0.970 |
| GPT-2 perplexity | 99.7 → 106.5 (×1.13) |
| Where edits land | 51% question · **13% correct option** · 37% wrong options |

The 13% is the bar to compare against. If the protected edits pile onto the correct option far more often, the surrogate has rediscovered the "answer hint" shortcut from Li et al. That is a headline analysis finding either way.

## 3. A problem we found and how we handled it
Embedding neighbours mix synonyms with **opposites**: `dry → wet`, `increase → decrease`, `larger → smaller`. SBERT ≥ 0.9 does not catch them. In the smoke test the search turned the correct answer "dry palms" into "moist palms", which flips the science. A protected set built that way would just be *wrong data*, which a grader would rightly attack.

**Fix:**
- Negation words (`not`, `never`, `doesnt`, …) can't be edited and can't be inserted.
- A swap is blocked if any WordNet synonym of one word is an antonym of the other. That catches `increase→reduce` and `larger→smaller`, which plain antonym lookup misses.

**Remaining limitation (goes in the report):** related words with different meanings still pass, e.g. `hot→cool`, `Earth→Moon`, `increase→decline`. A synonyms-only filter would stop them but shrink the neighbour lists so much that the attack likely stops working. Plan: hand-label ~50 sampled edits from Protected and from Random as meaning-kept or meaning-changed, and report the rate.

## 4. Next steps
| # | Task | Where | Notes |
|---|---|---|---|
| 1 | rsync `ue-arc/` to Sol (skip `.venv`). On the login node: `huggingface-cli login`, download WordNet, download both models into `/scratch/$USER/hf` | Sol login node | Commands in `ue-arc/README.md` |
| 2 | `mkdir -p logs && sbatch sol_baselines.sh` | Sol GPU (~2–3 h) | Zero-shot + clean fine-tune for both models × 3 seeds, then a 16-example protection test on real Llama-3-8B |
| 3 | **Check the Qwen3-8B ceiling risk.** If the clean fine-tune gains < ~3 points over zero-shot (within seed noise), email the TA about switching to `Qwen/Qwen3-8B-Base` | Us | The guideline expects fine-tuning to raise accuracy above zero-shot |
| 4 | `sbatch sol_protect.sh` | Sol GPU (~3–4 h) | Writes `protected/r0..r5.jsonl`, the loss curve, and `final.jsonl` |
| 5 | `sbatch sol_victims.sh` | Sol GPU (~4–5 h) | Random + Protected × 2 models × 3 seeds |
| 6 | `python summarize.py results` and `python metrics.py data/train.jsonl protected/r*.jsonl data/random_k2.jsonl` | Anywhere | Results table + trade-off plot (one point per round) |
| 7 | Design checkpoint (1 page), then the 6–8 page report (NeurIPS/ICLR style) | Us, while jobs run | Required sections are on pp. 6–7 of the guideline |
| 8 | Bonus idea | Only if 1–6 are done by Nov 20–21 | |

## 5. Where things are
```
Project_Guideline_1.pdf     assignment (results table on p.5, report sections on pp.6-7)
ue-arc/
  README.md                 setup, dry run, all three Sol bookings, off-Sol instructions
  data.py eval.py train_lora.py           baseline (prompt format, eval, LoRA training)
  protect.py random_edits.py metrics.py summarize.py
  sol_baselines.sh sol_protect.sh sol_victims.sh
  data/                     ARC splits + random_k2.jsonl
  protected/neighbours.json
```
Local Python env: `ue-arc/.venv` (made with uv). Install with `pip install -r requirements.txt`.
**Never commit an HF token.** Use `huggingface-cli login` or the `HF_TOKEN` env var.
