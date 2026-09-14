#!/usr/bin/env python3
"""Dump teacher prefixes (base) or chat traces (instruct) to JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_common import BASE_ID, INSTRUCT_ID, kaggle_out, load_spark, pick_device


def iter_base_texts(n: int):
    from datasets import load_dataset

    texts = []
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    for row in wiki:
        t = (row.get("text") or "").strip()
        if len(t) > 80:
            texts.append(t)
        if len(texts) >= n:
            return texts
    try:
        gsm = load_dataset("gsm8k", "main", split="train")
        for row in gsm:
            texts.append(row["question"])
            if len(texts) >= n:
                return texts
    except Exception as e:
        print("gsm8k skip", e, flush=True)
    return texts[:n]


def iter_instruct_prompts(n: int):
    from datasets import load_dataset

    prompts = []
    gsm = load_dataset("gsm8k", "main", split="train")
    for row in gsm:
        prompts.append(row["question"])
        if len(prompts) >= n:
            return prompts
    return prompts


def dump_base(tok, n: int, max_len: int, out: Path):
    texts = iter_base_texts(n)
    n_ok = 0
    with out.open("w") as f:
        for i, text in enumerate(texts):
            ids = tok(text, truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]
            if len(ids) < 16:
                continue
            rec = {"id": i, "teacher": "base", "input_ids": ids}
            f.write(json.dumps(rec) + "\n")
            n_ok += 1
    return n_ok


def dump_instruct(tok, model, device, n: int, max_new: int, max_prompt: int, out: Path):
    prompts = iter_instruct_prompts(n)
    n_ok = 0
    with out.open("w") as f:
        for i, q in enumerate(prompts):
            messages = [{"role": "user", "content": q}]
            try:
                prompt = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
            except TypeError:
                prompt = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": True},
                )
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_prompt)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=50,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            input_ids = gen[0].tolist()
            rec = {
                "id": i,
                "teacher": "instruct",
                "prompt": q,
                "n_in": int(inputs["input_ids"].shape[1]),
                "input_ids": input_ids,
            }
            f.write(json.dumps(rec) + "\n")
            n_ok += 1
            if (i + 1) % 10 == 0:
                print(f"dumped {i+1}/{len(prompts)}", flush=True)
    return n_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", choices=["base", "instruct"], required=True)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--out", default="teacher_dump.jsonl")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    out = Path(kaggle_out(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_id = BASE_ID if args.teacher == "base" else INSTRUCT_ID

    tok, model = load_spark(model_id, device, dtype=dtype, eval_mode=True)
    if args.teacher == "base":
        n_ok = dump_base(tok, args.n, args.max_len, out)
    else:
        n_ok = dump_instruct(tok, model, device, args.n, args.max_new, args.max_len, out)

    summary = {"ok": n_ok > 0, "teacher": args.teacher, "n": n_ok, "out": str(out)}
    Path(kaggle_out("dump_summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if n_ok == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
