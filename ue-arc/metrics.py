"""Stealthiness metrics: GPT-2 perplexity and SBERT cosine similarity. Used as the edit filters and in the report.

python metrics.py data/train.jsonl protected/final.jsonl data/random_k2.jsonl
  -> per file: mean SBERT, mean PPL ratio, edits/example, where the edits landed (question / correct / wrong option)
"""
import json
import math
import sys

import torch

from data import load_jsonl

SBERT = "sentence-transformers/all-MiniLM-L6-v2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_models = {}


def ex_text(ex):
    """x = question + choices: the text the constraints are measured on."""
    return " ".join([ex["question"]] + ex["choices"])


@torch.no_grad()
def ppl(texts):
    if "gpt2" not in _models:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _models["gpt2"] = (AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval(),
                           AutoTokenizer.from_pretrained("gpt2"))
    model, tok = _models["gpt2"]
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids[:, :1024].to(DEV)
        out.append(math.exp(model(ids, labels=ids).loss.item()))
    return out


@torch.no_grad()
def sbert_sim(a, b):
    if "sbert" not in _models:
        from sentence_transformers import SentenceTransformer
        _models["sbert"] = SentenceTransformer(SBERT, device=DEV)
    m = _models["sbert"]
    ea = m.encode(a, convert_to_tensor=True, normalize_embeddings=True)
    eb = m.encode(b, convert_to_tensor=True, normalize_embeddings=True)
    return (ea * eb).sum(-1).tolist()


def passes(orig, cands, min_sim=0.9, max_ppl_ratio=1.2):
    """Design constraints, always measured against the ORIGINAL example (so they bound all k edits together)."""
    o, cs = ex_text(orig), [ex_text(c) for c in cands]
    base = ppl([o])[0]
    return [s >= min_sim and p <= max_ppl_ratio * base
            for s, p in zip(sbert_sim([o] * len(cs), cs), ppl(cs))]


if __name__ == "__main__":
    orig = {r["id"]: r for r in load_jsonl(sys.argv[1])}
    for path in sys.argv[2:]:
        rows = load_jsonl(path)
        o, e = [ex_text(orig[r["id"]]) for r in rows], [ex_text(r) for r in rows]
        po, pe = ppl(o), ppl(e)
        ratios = [b / a for a, b in zip(po, pe)]
        where = {"question": 0, "correct": 0, "wrong": 0}
        for r in rows:
            for ed in r.get("edits", []):
                f = ed["field"]
                where["question" if f == 0 else "correct" if f == r["answer_idx"] + 1 else "wrong"] += 1
        n_edits = sum(where.values())
        print(json.dumps({"file": path, "n": len(rows),
                          "sbert": sum(sbert_sim(o, e)) / len(rows),
                          "ppl_orig": sum(po) / len(rows), "ppl": sum(pe) / len(rows),
                          "ppl_ratio": sum(ratios) / len(rows),
                          "edits_per_ex": n_edits / len(rows),
                          "edit_share": {k: v / max(1, n_edits) for k, v in where.items()}}))
