"""Min-min protection (Huang et al. 2021), text version: k whole-word swaps per example chosen to MINIMISE the
surrogate's answer-letter loss, alternating with surrogate LoRA training on the current protected set.

python protect.py --model meta-llama/Meta-Llama-3-8B --train data/train.jsonl --out protected
  -> protected/r{0..R}.jsonl (one protected set per round), protected/ckpt_r{R}/, protected/rounds.jsonl,
     protected/neighbours.json, protected/final.jsonl (= last round)
Resumable: rounds whose r{R}.jsonl exists are skipped; a saved ckpt_r{R} skips that round's training.
"""
import argparse
import functools
import json
import os
import re
import shutil

import torch
import torch.nn.functional as F

from data import format_prompt, load_jsonl, save_jsonl
from eval import evaluate, load_model
from metrics import passes
from train_lora import add_lora, collate, encode, set_seed, train

WORD = re.compile(r"[A-Za-z]+")


def fields(ex):
    """Field 0 = question, field i+1 = choice i."""
    return [ex["question"]] + ex["choices"]


def swap(ex, f, w, new):
    """Copy of ex with the w-th word of field f replaced by new."""
    t = fields(ex)[f]
    m = list(WORD.finditer(t))[w]
    t = t[:m.start()] + new + t[m.end():]
    ex = dict(ex, choices=list(ex["choices"]))
    if f == 0:
        ex["question"] = t
    else:
        ex["choices"][f - 1] = t
    return ex


def field_starts(ex):
    """Char offset of each field inside format_prompt(ex)."""
    starts, pos = [len("Question: ")], len("Question: ") + len(ex["question"]) + 1
    for c in ex["choices"]:
        starts.append(pos + 3)  # "A. "
        pos += 3 + len(c) + 1
    prompt = format_prompt(ex)
    assert all(prompt[s:s + len(t)] == t for s, t in zip(starts, fields(ex))), "format_prompt changed"
    return starts


def slots(ex, tok):
    """Editable words: (field, word index, word, token position) for words that are ONE token ' word' in the prompt."""
    enc = tok(format_prompt(ex), return_offsets_mapping=True)
    end2tok = {e: i for i, (_, e) in enumerate(enc["offset_mapping"])}
    out = []
    for f, (start, text) in enumerate(zip(field_starts(ex), fields(ex))):
        for w, m in enumerate(WORD.finditer(text)):
            i = end2tok.get(start + m.end())
            if i is not None and enc["input_ids"][i] == word_id(tok, m.group()):
                out.append((f, w, m.group(), i))
    return out


@functools.lru_cache(maxsize=None)
def word_id(tok, word):
    """Token id of ' word', or None if it is not a single token."""
    ids = tok.encode(" " + word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


# never edit, and never introduce, a negation: flipping one flips the question's meaning
NEGATIONS = {"not", "no", "never", "nor", "none", "nothing", "neither", "nobody", "nowhere", "without", "cannot",
             "isnt", "arent", "wasnt", "werent", "dont", "doesnt", "didnt", "wont", "cant", "couldnt", "shouldnt",
             "wouldnt", "hasnt", "havent", "hadnt"}


@functools.lru_cache(maxsize=None)
def synonyms(w):
    """w plus every lemma of every WordNet sense of w (synsets() also matches inflections: 'increases')."""
    from nltk.corpus import wordnet as wn
    return {w} | {l.name().lower() for s in wn.synsets(w) for l in s.lemmas()}


@functools.lru_cache(maxsize=None)
def antonyms(w):
    from nltk.corpus import wordnet as wn
    return {a.name().lower() for s in wn.synsets(w) for l in s.lemmas() for a in l.antonyms()}


def opposite(w, c):
    """c means roughly the opposite of w: one's antonym is a synonym of the other (catches increase -> reduce,
    larger -> smaller, which direct WordNet antonym pairs miss)."""
    return bool(antonyms(w) & synonyms(c) or antonyms(c) & synonyms(w))


def build_neighbours(emb, tok, words, k=50):
    """Closest input-embedding neighbours (cosine) among single-token whole words with the same capitalisation,
    minus WordNet antonyms and negations (embedding neighbours mix synonyms with opposites: dry -> wet).
    emb = the surrogate's input embedding matrix (vocab x dim)."""
    vocab = {}
    for i in range(len(tok)):
        s = tok.decode([i])
        if s.startswith(" ") and len(s) > 2 and s[1:].isascii() and s[1:].isalpha():
            vocab[i] = s[1:]
    vids = list(vocab)
    E = F.normalize(emb.float(), dim=-1)
    V = E[torch.tensor(vids, device=E.device)]
    out = {}
    for w in words:
        nb = []
        if w.lower() in NEGATIONS:
            out[w] = nb
            continue
        for j in (V @ E[word_id(tok, w)]).topk(min(500, len(vids))).indices.tolist():
            c = vocab[vids[j]]
            if (c.lower() != w.lower() and c.istitle() == w.istitle() and c.isupper() == w.isupper()
                    and c.lower() not in NEGATIONS and not opposite(w.lower(), c.lower())):
                nb.append(c)
                if len(nb) == k:
                    break
        out[w] = nb
    return out


@torch.no_grad()
def answer_losses(model, tok, rows, bs=16):
    """Exact answer-letter cross-entropy, one value per row."""
    out = []
    for i in range(0, len(rows), bs):
        ids, att, lab = (t.to(model.device) for t in collate([encode(tok, r) for r in rows[i:i + bs]], tok.pad_token_id))
        logits, lab = model(input_ids=ids, attention_mask=att).logits[:, :-1], lab[:, 1:]
        m = lab != -100
        assert (m.sum(1) == 1).all(), "expected exactly one answer token per row"
        out += F.cross_entropy(logits[m].float(), lab[m], reduction="none").tolist()
    return out


def embed_grad(model, tok, ex):
    """Gradient of the answer-letter loss w.r.t. the input embeddings (positions x dim), and the loss."""
    x, y = encode(tok, ex)
    emb = model.get_input_embeddings()(torch.tensor([x], device=model.device)).detach().requires_grad_()
    loss = model(inputs_embeds=emb, labels=torch.tensor([y], device=model.device)).loss
    return torch.autograd.grad(loss, emb)[0][0].float(), loss.item()


def search(model, tok, ex, nbrs, k=2, top=20):
    """Greedy k swaps. Each step: HotFlip shortlist by first-order loss change (e_new - e_old)^T grad,
    keep the `top` most negative, drop those failing the filters, exact re-score, take the best if it lowers the loss."""
    E = model.get_input_embeddings().weight
    cur, edits = ex, []
    g, loss = embed_grad(model, tok, cur)
    loss0 = loss
    for step in range(k):
        done = {(e["field"], e["word"]) for e in edits}
        scored = []
        for f, w, word, pos in slots(cur, tok):
            if (f, w) in done or not nbrs.get(word):
                continue
            new_ids = torch.tensor([word_id(tok, n) for n in nbrs[word]], device=E.device)
            delta = (E[new_ids].float() - E[word_id(tok, word)].float()) @ g[pos]
            scored += [(d, f, w, word, n) for d, n in zip(delta.tolist(), nbrs[word])]
        scored = sorted(scored)[:top]
        cands = [swap(cur, f, w, n) for _, f, w, _, n in scored]
        ok = passes(ex, cands) if cands else []
        keep = [(s, c) for s, c, o in zip(scored, cands, ok) if o]
        if not keep:
            break
        losses = answer_losses(model, tok, [c for _, c in keep])
        best = min(range(len(keep)), key=losses.__getitem__)
        if losses[best] >= loss:
            break
        (_, f, w, word, n), cur = keep[best]
        edits.append({"field": f, "word": w, "from": word, "to": n})
        loss = losses[best]
        if step < k - 1:
            g, _ = embed_grad(model, tok, cur)
    return dict(cur, edits=edits), loss0, loss


def protect_set(model, tok, rows, nbrs, k, top):
    model.eval()
    out, before, after = [], [], []
    for i, ex in enumerate(rows):
        p, l0, l1 = search(model, tok, ex, nbrs, k, top)
        out.append(p); before.append(l0); after.append(l1)
        if i % 100 == 0:
            print(f"  search {i}/{len(rows)} loss {l0:.4f} -> {l1:.4f} edits {p['edits']}", flush=True)
    return out, sum(before) / len(rows), sum(after) / len(rows)


def hub_embeddings(name):
    """Only the input embedding matrix, read from the checkpoint shard (no full model load: laptop-friendly)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    key = "model.embed_tokens.weight"
    with open(hf_hub_download(name, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][key]
    with safe_open(hf_hub_download(name, shard), "pt") as f:
        return f.get_tensor(key)


def load_lora(model, d):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model, load_file(os.path.join(d, "adapter_model.safetensors")))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--out", default="protected")
    p.add_argument("--k", type=int, default=2, help="edit budget per example")
    p.add_argument("--top", type=int, default=20, help="HotFlip shortlist size")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--train_epochs", type=int, default=2, help="surrogate epochs per round (~140 steps/epoch at bs 8)")
    p.add_argument("--stop_acc", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="dry runs only")
    p.add_argument("--neighbours_only", action="store_true", help="write neighbours.json from the embeddings and exit (CPU)")
    a = p.parse_args()

    rows = load_jsonl(a.train, a.limit)
    os.makedirs(a.out, exist_ok=True)
    nb_path = os.path.join(a.out, "neighbours.json")
    if a.neighbours_only:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        with open(nb_path, "w") as f:
            json.dump(build_neighbours(hub_embeddings(a.model), tok, sorted({s[2] for r in rows for s in slots(r, tok)})), f)
        print(f"wrote {nb_path}")
        raise SystemExit(0)

    set_seed(a.seed)
    model, tok = load_model(a.model)
    model = add_lora(model)
    pset = lambda r: os.path.join(a.out, f"r{r}.jsonl")
    ckpt = lambda r: os.path.join(a.out, f"ckpt_r{r}")

    def log(rec):
        print(json.dumps(rec), flush=True)
        with open(os.path.join(a.out, "rounds.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    if not os.path.exists(nb_path):
        words = sorted({s[2] for r in rows for s in slots(r, tok)})
        with open(nb_path, "w") as f:
            json.dump(build_neighbours(model.get_input_embeddings().weight, tok, words), f)
    with open(nb_path) as f:
        nbrs = json.load(f)

    last = None
    for r in range(a.rounds + 1):
        if os.path.exists(pset(r)):
            last = r
            continue
        if r > 0:  # surrogate step: theta_r = train theta_{r-1} on the round r-1 protected set
            prev = load_jsonl(pset(r - 1))
            if os.path.isdir(ckpt(r)):
                load_lora(model, ckpt(r))
            else:
                if r > 1:
                    load_lora(model, ckpt(r - 1))  # no-op unless resuming
                tl = train(model, tok, prev, epochs=a.train_epochs, seed=a.seed + r,
                           log_path=os.path.join(ckpt(r), "loss.jsonl"))  # per-step surrogate loss curve
                model.save_pretrained(ckpt(r))
                log({"round": r, "phase": "train", "train_loss_last20": sum(x["loss"] for x in tl[-20:]) / len(tl[-20:])})
            acc = evaluate(model, tok, prev)
            log({"round": r, "phase": "eval", "surrogate_train_acc": acc})
            if acc >= a.stop_acc:
                break
        prot, l0, l1 = protect_set(model, tok, rows, nbrs, a.k, a.top)
        save_jsonl(prot, pset(r))
        log({"round": r, "phase": "search", "clean_loss": l0, "protected_loss": l1,
             "edits_per_ex": sum(len(x["edits"]) for x in prot) / len(prot)})
        last = r
    shutil.copy(pset(last), os.path.join(a.out, "final.jsonl"))
    print(f"final = round {last}")
