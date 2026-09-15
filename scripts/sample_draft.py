#!/usr/bin/env python3
"""Generate short continuations from a Spark draft checkpoint."""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_common import fix_generation_config, kaggle_out, patch_rope_validation

PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a small village",
    "def fibonacci(n):\n    ",
    "Question: What is 12 * 3?\nAnswer:",
    "In 1969, NASA astronauts",
    "Python is a programming language that",
]


def resolve_draft(raw: str) -> str:
    if raw and Path(raw).exists():
        return raw
    for pat in (
        "/kaggle/input/**/spark-x25-draft-0.5B-base-kd/config.json",
        "/kaggle/input/**/spark-x25-draft-0.5B-instruct-kd/config.json",
    ):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return str(Path(hits[0]).parent)
    raise SystemExit("no draft checkpoint found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft-path", default="")
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--out", default="draft_samples.json")
    args = p.parse_args()

    path = resolve_draft(args.draft_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print("draft", path, "device", device, flush=True)
    patch_rope_validation()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=dtype, device_map=None
    )
    model.to(device)
    model.eval()
    fix_generation_config(model)

    rows = []
    for i, prompt in enumerate(PROMPTS):
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        t0 = time.time()
        with torch.no_grad():
            greedy = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            sampled = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        dt = time.time() - t0
        gtxt = tok.decode(greedy[0], skip_special_tokens=True)
        stxt = tok.decode(sampled[0], skip_special_tokens=True)
        row = {"id": i, "prompt": prompt, "greedy": gtxt, "sampled": stxt, "seconds": round(dt, 2)}
        rows.append(row)
        print("=" * 60, flush=True)
        print("PROMPT:", prompt, flush=True)
        print("GREEDY:", gtxt, flush=True)
        print("SAMPLED:", stxt, flush=True)

    out = Path(kaggle_out(args.out))
    out.write_text(json.dumps({"ok": True, "draft": path, "device": str(device), "rows": rows}, indent=2))
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
