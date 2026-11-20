"""Bonus: model-agnostic "answer hint" protection (Li et al. 2023, Sec. 6, adapted to ARC + LLM fine-tuning).

Appends one marker to the CORRECT choice only. "!" was chosen because no ARC choice (train/val/test, ~10k) ends
with it, so it is a clean shortcut; it mirrors Li et al.'s "!" label hint. One edit per question, no gradients.
Hypothesis (docs/bonus_hypothesis.md, committed before results): see that file.

  python make_hint.py --train data/train.jsonl --out data/hint.jsonl [--marker "!"]
"""
import argparse
import json

from data import load_jsonl


def add_hint(ex, marker="!"):
    out = dict(ex, choices=list(ex["choices"]))
    i = ex["answer_idx"]
    old = out["choices"][i]
    if not old.rstrip().endswith(marker):
        out["choices"][i] = old.rstrip() + marker
    out["edits"] = [{"field": i + 1, "word": -1, "from": "", "to": marker, "kind": "append"}]
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--out", default="data/hint.jsonl")
    p.add_argument("--marker", default="!")
    a = p.parse_args()
    rows = [add_hint(r, a.marker) for r in load_jsonl(a.train)]
    for r in rows:  # sanity: answer unchanged, only the correct choice changed, exactly one marker
        assert sum(c.endswith(a.marker) for c in r["choices"]) == 1, r["id"]
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {a.out}; example correct choice: {rows[0]['choices'][rows[0]['answer_idx']]!r}")
