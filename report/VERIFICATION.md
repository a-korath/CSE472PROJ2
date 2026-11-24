# Final verification: CSE 472 Project 2.5 report (Mon Nov 23)

Checked against the mentor's guideline, the class announcement (Nov 2), the cited papers and the run files.
✅ = verified, with the evidence in the right-hand column. Fixes made in this pass are marked **(fixed)**.

## 1. Requirements (guideline + class)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| R1 | 6–8 pages, ML conference template | ✅ | 8 pages (about 6.3 body + references), ICLR 2027 style |
| R2 | PDF | ✅ | `main.pdf` |
| R3 | Class sections: Abstract, Introduction, Related Works, Model Description, Experiment, Future Work, References | ✅ | `\section` titles |
| R4 | Research question + one-sentence answer | ✅ | §1, "Research question" / "One-sentence answer" |
| R5 | Related work: the two papers + others used | ✅ | §2: Huang 2021, Li & Liu 2023, survey, VTG, TextCloak, When Priors Backfire, Geirhos |
| R6 | Method: bilevel formulation, loss, surrogate, perturbation, and why | ✅ | §3, Eq. 1, "Why:" lines |
| R7 | Setup: data, models, LoRA config, metrics, seeds | ✅ | §4.1 |
| R8 | Results table, per victim, 3 seeds mean ± std, accuracy / GPT-2 PPL / SBERT | ✅ | Table 1 |
| R9 | White-box = Llama-3-8B, black-box = Qwen3-8B | ✅ | §4.1, Table 1 (Qwen3-8B-Base marked as an added victim) |
| R10 | Random-edit control, same budget | ✅ | §3 "Control" (same words, neighbours, k = 2) |
| R11 | Protected data + surrogate training-loss curve | ✅ | `ue-arc/protected/`, Fig. 2 |
| R12 | Trade-off plot + examples | ✅ | Fig. 1, Table 2 |
| R13 | Analysis & limitations: what transferred, what didn't, why | ✅ | §4.3 |
| R14 | "What to expect" addressed honestly | ✅ | §4.2–4.3 (protected < clean, but not < random) |
| R15 | Bonus: hypothesis first, outcome reported even if negative | ✅ | §4.4; `b554aff` (11:58) is before the first hint result (14:54) |
| R16 | Bonus uses the same runs and metrics | ✅ | Hint rows in Table 1 and Fig. 1, 3 seeds, all victims |
| R17 | Every paper, package, model, dataset and tool cited | ✅ **(fixed)** | Added `huggingface_hub` and `safetensors` (both imported in our code); every import is now covered |
| R18 | Generative-AI use acknowledged | ✅ | Acknowledgments (Arjun to confirm his tools) |
| R19 | Both checkpoints labelled | ✅ | "(Design Checkpoint)" §3, "(Results Checkpoint)" §4.2 |
| — | Hand-labelling | Not required | The guideline measures meaning with SBERT only. §4.3 now states it as future work **(fixed wording)** |

## 2. Citations (28 entries)

**Mechanical checks**
- Every bib key is cited, and every citation resolves. No duplicates. BibTeX: 0 warnings.

**Claims checked against the paper text**

| Paper | Claim in our report | Status | Source location |
|---|---|---|---|
| Huang et al., ICLR 2021 | Bilevel min–min problem | ✅ | Eq. 2 |
| Huang et al., ICLR 2021 | Sample-wise and class-wise noise | ✅ | §3.1 |
| Huang et al., ICLR 2021 | Class-wise is "particularly more effective" | ✅ | §4.2 |
| Huang et al., ICLR 2021 | Stop once error on perturbed training data < λ (0.01) | ✅ | §3.2, App. B |
| Li & Liu, TrustNLP 2023 (pp. 249–259) | Authors and venue | ✅ | ACL Anthology |
| Li & Liu, TrustNLP 2023 | LSTM/BiDAF surrogates; cos ≥ 0.5; one substitution | ✅ | Paper text |
| Li & Liu, TrustNLP 2023 | Eq. 5 form; Error-min-0 Jaccard 0; top-5 share 60–100% | ✅ | Paper text |
| Li & Liu, TrustNLP 2023 | AG-News Error-min-1 = 0.08; label/answer hints; stop on test metrics | ✅ | Paper text |
| TextCloak (Zhao et al. 2026, arXiv 2607.28862) | Authors | ✅ | arXiv page |
| TextCloak | Eq. 4 uses the difference form | ✅ | Eq. 4 |
| TextCloak | Pretraining-priors argument | ✅ | Sec. 5.2.1 |
| TextCloak | Textual UE on ARC within 2.6 points | ✅ | Table 3 |
| TextCloak | Qwen3-8B 78.7 → 91.9 | ✅ | Table 2 |
| TextCloak | Same splits (1,119 / 299 / 1,172) | ✅ | Table 6 |
| When Priors Backfire, ICLR 2026 | Authors | ✅ | Paper |
| When Priors Backfire, ICLR 2026 | Image classification only | ✅ | Paper |
| When Priors Backfire, ICLR 2026 | Pretrained models learn genuine features despite UE shortcuts | ✅ | Abstract, Fig. 1 |
| Survey (arXiv 2503.23536) | Authors | ✅ | Paper |
| VTG, NeurIPS 2025 | Authors and venue | ✅ | Paper |

**Other entries**
- HotFlip, Alzantot, Geirhos, ARC, LoRA, Llama 3, Qwen3, GPT-2, SBERT, MiniLM, WordNet, NLTK, Transformers, PEFT,
  PyTorch, Datasets, Accelerate, SciPy, Matplotlib and Sol: metadata matches the standard published records.
- Added the NLTK publisher **(fixed)**.
- "and others" is used only for 100+-author reports and large library papers; the first authors are correct.
- The TextCloak title follows arXiv v2. The guideline lists an older title.
- All prose is our own wording. The only quotes are our own pre-registered hypothesis and short labels.

## 3. Numbers

**Table**
- `make_figures.py` regenerates `table_main.tex`, and the result is byte-identical to the committed copy.

**Welch t-tests (Llama-3-8B)**
- Protected vs random: p = 0.829
- Round 0 vs random: p = 0.682
- Protected vs clean: p = 0.021
- Random vs clean: p = 0.0046
- All four match the text (0.83, 0.68, 0.02, 0.005).

**Surrogate (`rounds.jsonl`)**
- Loss on unedited questions: 0.213–0.537 in rounds 1–5. Text: 0.21–0.54 ✅
- Round 5 loss: protected 0.0068 vs clean 0.213 ✅
- Surrogate training accuracy: 99.2%, then ≥ 99.7% ✅
- Edits per question: 1.18 ✅

**Hint**
- Llama-3-8B seeds: 42.49–64.93 ✅
- Qwen3-8B-Base seeds: 72.01–90.36 ✅
- Every seed is below that model's zero-shot (max 64.9 < 77.5, 90.9 < 91.0, 90.4 < 92.4) ✅
- Qwen3-8B: −1.14 vs zero-shot ✅
- Llama-3-8B: −26.0 vs zero-shot ✅

**Gains (clean − zero-shot)**
- Llama-3-8B +5.15, Qwen3-8B +1.82, Qwen3-8B-Base +0.23 ✅

## 4. Document

- **Clean rebuild** (`latexmk -C && latexmk -pdf`): 0 errors, 0 undefined references, 0 overfull boxes, 0 BibTeX warnings.
- **Visual check:** all 8 pages inspected. No "??", figures readable, tables intact, the header shows the course name.
- **Spell check:** no typos. The flagged words are technical terms. British spelling ("neighbour") is used consistently.
- **PDF metadata:** title, authors and subject are set **(fixed)**.
- **Overleaf zip:** 10 files. It compiles to 8 pages with text identical to `main.pdf`.

## 5. Left for the humans

1. Arjun confirms the AI-use statement covers every tool he used.
2. Decide who uploads to Canvas (group or individual), and open the file after uploading.
3. Read the TextCloak sentences (§2, §4.3 Limitations) once more, since the mentor wrote that paper.
