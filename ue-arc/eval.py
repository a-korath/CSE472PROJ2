"""Accuracy on ARC by comparing the model's next-token scores for ' A', ' B', ' C', ...

Zero-shot:   python eval.py --model meta-llama/Meta-Llama-3-8B --out results/llama_zeroshot.json
With LoRA:   python eval.py --model ... --adapter adapters/llama_clean_s1 --out ...
Raw prompts, no chat template -> identical format for every model (Qwen3 thinking mode never triggers).
"""
import argparse
import json
import os

import torch

from data import LETTERS, format_prompt, load_jsonl


def load_model(name, adapter=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kw = {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}  # arg was renamed
    # no CUDA (laptop): plain CPU; "auto" there offloads to meta/disk and breaks backward
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto" if torch.cuda.is_available() else None, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model, tok


def letter_ids(tok):
    ids = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]
    assert len(set(ids)) == len(LETTERS), "letter tokens collide; check tokenizer"
    return ids


@torch.no_grad()
def evaluate(model, tok, rows, bs=16):
    model.eval()
    ids = letter_ids(tok)
    old_side, tok.padding_side = tok.padding_side, "left"  # last position = last real token
    correct = 0
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        enc = tok([format_prompt(r) for r in batch], return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        for j, r in enumerate(batch):
            n = len(r["choices"])  # only score letters that exist for this question
            pred = int(torch.argmax(logits[j, ids[:n]]))
            correct += int(pred == r["answer_idx"])
    tok.padding_side = old_side
    return correct / len(rows)


def write_result(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--data", default="data/test.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="dry runs only")
    a = p.parse_args()

    model, tok = load_model(a.model, a.adapter)
    rows = load_jsonl(a.data, a.limit)
    acc = evaluate(model, tok, rows)
    write_result(a.out, {"model": a.model, "adapter": a.adapter, "data": a.data,
                         "n": len(rows), "accuracy": acc})
