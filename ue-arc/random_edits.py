"""Random-edit control: same k swaps, but at random positions with random candidates from the SAME neighbour lists,
through the SAME filters. No surrogate model needed (tokenizer only).

python random_edits.py --tokenizer meta-llama/Meta-Llama-3-8B --neighbours protected/neighbours.json --out data/random_k2.jsonl
"""
import argparse
import json
import random

from data import load_jsonl, save_jsonl
from metrics import passes
from protect import slots, swap


def random_edit(ex, tok, nbrs, rng, k=2, batch=8, batches=5):
    cur, edits = ex, []
    for _ in range(k):
        done = {(e["field"], e["word"]) for e in edits}
        pool = [(f, w, word) for f, w, word, _ in slots(cur, tok) if (f, w) not in done and nbrs.get(word)]
        if not pool:
            break
        for _ in range(batches):  # try random (slot, neighbour) pairs until one passes the filters
            picks = [(f, w, word, rng.choice(nbrs[word])) for f, w, word in (rng.choice(pool) for _ in range(batch))]
            cands = [swap(cur, f, w, n) for f, w, _, n in picks]
            hit = next(((pk, c) for pk, c, ok in zip(picks, cands, passes(ex, cands)) if ok), None)
            if hit:
                (f, w, word, n), cur = hit
                edits.append({"field": f, "word": w, "from": word, "to": n})
                break
    return dict(cur, edits=edits)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="surrogate tokenizer, so editable words match protect.py")
    p.add_argument("--neighbours", required=True)
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    with open(a.neighbours) as f:
        nbrs = json.load(f)
    rng = random.Random(a.seed)
    out = []
    for i, ex in enumerate(load_jsonl(a.train, a.limit)):
        out.append(random_edit(ex, tok, nbrs, rng, a.k))
        if i % 100 == 0:
            print(f"{i}: {out[-1]['edits']}", flush=True)
    save_jsonl(out, a.out)
    print(f"edits/example: {sum(len(r['edits']) for r in out) / len(out):.2f}")
