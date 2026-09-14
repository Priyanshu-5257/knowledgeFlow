#!/usr/bin/env python3
"""Stage-1/2 DistillSpec: CE + chunked reverse-KL of a 4-layer Spark draft."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_draft import copy_draft_weights, parse_layer_map
from spark_common import (
    BASE_ID,
    DEFAULT_LAYER_MAP,
    DRAFT_LAYER_TYPES,
    INSTRUCT_ID,
    count_params,
    fix_generation_config,
    kaggle_out,
    patch_rope_validation,
    teacher_student_devices,
)


def chunked_reverse_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Mean KL(teacher || student) over batch and time, chunked on vocab."""
    t = teacher_logits.float()
    s = student_logits.float()
    t_lse = torch.logsumexp(t, dim=-1)
    s_lse = torch.logsumexp(s, dim=-1)
    vocab = t.shape[-1]
    kl = torch.zeros(t.shape[:-1], device=s.device, dtype=torch.float32)
    for start in range(0, vocab, chunk):
        t_c = t[..., start : start + chunk]
        s_c = s[..., start : start + chunk]
        t_logp = (t_c - t_lse.unsqueeze(-1)).clamp(min=-30)
        s_logp = (s_c - s_lse.unsqueeze(-1)).clamp(min=-30)
        kl = kl + (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
    return kl.mean()


def token_match(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> float:
    t = teacher_logits[..., :-1, :].argmax(dim=-1)
    s = student_logits[..., :-1, :].argmax(dim=-1)
    return float((t == s).float().mean().item())


def packed_base_sequences(tok, n: int, max_len: int) -> list[list[int]]:
    from datasets import load_dataset

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    buf: list[int] = []
    seqs: list[list[int]] = []
    for row in wiki:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        while len(buf) >= max_len:
            seqs.append(buf[:max_len])
            buf = buf[max_len:]
            if len(seqs) >= n:
                return seqs
    if buf and len(seqs) < n:
        pad_id = tok.pad_token_id or tok.eos_token_id or 0
        seqs.append((buf + [pad_id] * max_len)[:max_len])
    return seqs[:n]


def load_jsonl_ids(path: Path, max_len: int) -> list[list[int]]:
    seqs = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            ids = rec["input_ids"][:max_len]
            if len(ids) >= 16:
                seqs.append(ids)
    return seqs


def instantiate_draft(teacher, teacher_id: str, device, dtype, layer_map):
    cfg = AutoConfig.from_pretrained(teacher_id, trust_remote_code=True)
    cfg.num_hidden_layers = 4
    cfg.layer_types = list(DRAFT_LAYER_TYPES)
    draft = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=dtype)
    draft.to(device)
    report = copy_draft_weights(teacher, draft, layer_map)
    if report["load_missing"]:
        raise SystemExit(f"incomplete copy {report['load_missing'][:8]}")
    fix_generation_config(draft)
    print("built draft params", count_params(draft), "copied", report["copied"], flush=True)
    return draft


def load_draft(path: str, device, dtype):
    patch_rope_validation()
    draft = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=dtype, device_map=None
    )
    draft.to(device)
    fix_generation_config(draft)
    print("loaded draft", path, "params", count_params(draft), flush=True)
    return draft


def freeze_embeddings(model, freeze: bool) -> None:
    for name, param in model.named_parameters():
        if "embedding" in name or "lm_head" in name:
            param.requires_grad_(not freeze)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["base", "instruct"], default="base")
    p.add_argument("--teacher", default="")
    p.add_argument("--draft-path", default="")
    p.add_argument("--data", default="")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--kl-chunk", type=int, default=4096)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--freeze-embed-steps", type=int, default=10**9)
    p.add_argument("--layer-map", default=",".join(str(i) for i in DEFAULT_LAYER_MAP))
    p.add_argument("--out", default="spark-x25-draft-0.5B-base-kd")
    args = p.parse_args()

    teacher_id = args.teacher or (BASE_ID if args.stage == "base" else INSTRUCT_ID)
    t_dev, s_dev = teacher_student_devices()
    teacher_dtype = torch.float16 if t_dev.type == "cuda" else torch.float32
    student_dtype = torch.float32
    layer_map = parse_layer_map(args.layer_map)
    out_dir = Path(kaggle_out(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("devices teacher", t_dev, "student", s_dev, "stage", args.stage, flush=True)
    patch_rope_validation()
    tok = AutoTokenizer.from_pretrained(teacher_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id, trust_remote_code=True, torch_dtype=teacher_dtype, device_map=None
    )
    teacher.to(t_dev)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    fix_generation_config(teacher)
    print("teacher params", count_params(teacher), flush=True)

    if args.draft_path:
        student = load_draft(args.draft_path, s_dev, student_dtype)
    else:
        student = instantiate_draft(teacher, teacher_id, s_dev, student_dtype, layer_map)

    freeze_embeddings(student, freeze=True)
    student.train()
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        student.config.use_cache = False

    if args.data:
        seqs = load_jsonl_ids(Path(args.data), args.seq_len)
    else:
        print("packing wikitext sequences", args.n, "x", args.seq_len, flush=True)
        seqs = packed_base_sequences(tok, args.n, args.seq_len)
    if not seqs:
        raise SystemExit("no training sequences")
    print("n_seq", len(seqs), "len0", len(seqs[0]), flush=True)
    dump_path = Path(kaggle_out("teacher_base_packed.jsonl"))
    with dump_path.open("w") as f:
        for i, ids in enumerate(seqs):
            f.write(json.dumps({"id": i, "teacher": args.stage, "input_ids": ids}) + "\n")

    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    history = []
    t0 = time.time()
    student.zero_grad(set_to_none=True)
    skipped = 0

    for step in range(1, args.steps + 1):
        ids = seqs[(step - 1) % len(seqs)]
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        t_x = x.to(t_dev)
        s_x = x.to(s_dev)
        attn = torch.ones_like(t_x)

        with torch.no_grad():
            t_out = teacher(input_ids=t_x, attention_mask=attn, use_cache=False)
            t_logits = t_out.logits
        s_out = student(input_ids=s_x, attention_mask=attn.to(s_dev), use_cache=False)
        s_logits = s_out.logits

        t_logits_s = t_logits.to(s_dev)
        ce = F.cross_entropy(
            s_logits[:, :-1].float().reshape(-1, s_logits.size(-1)),
            s_x[:, 1:].reshape(-1),
        )
        kl = chunked_reverse_kl(t_logits_s[:, :-1], s_logits[:, :-1], chunk=args.kl_chunk)
        loss = ce + args.beta * kl
        if not torch.isfinite(loss):
            skipped += 1
            student.zero_grad(set_to_none=True)
            print("skip non-finite loss at step", step, flush=True)
            history.append({"step": step, "loss": None, "ce": None, "kl": None, "match": 0.0, "skipped": True})
            continue
        (loss / args.grad_accum).backward()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step == args.freeze_embed_steps:
            freeze_embeddings(student, freeze=False)
            trainable = [p for p in student.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=args.lr)
            print("unfroze embeddings at step", step, flush=True)

        with torch.no_grad():
            match = token_match(t_logits_s, s_logits)
        row = {
            "step": step,
            "loss": float(loss.detach().item()),
            "ce": float(ce.detach().item()),
            "kl": float(kl.detach().item()),
            "match": match,
        }
        history.append(row)
        log_every = 50 if args.steps >= 500 else 10
        if step == 1 or step % log_every == 0 or step == args.steps:
            print(row, "elapsed", round(time.time() - t0, 1), flush=True)

        del t_out, s_out, t_logits, s_logits, t_logits_s, loss, ce, kl
        if t_dev.type == "cuda":
            torch.cuda.empty_cache()

    student.eval()
    tok.save_pretrained(out_dir)
    student.save_pretrained(out_dir)
    finite = [r for r in history if isinstance(r.get("loss"), float) and r["loss"] == r["loss"]]
    first, last = (finite[0], finite[-1]) if finite else (history[0], history[-1])
    improved = False
    if finite and isinstance(first.get("loss"), float) and isinstance(last.get("loss"), float):
        improved = last["loss"] < first["loss"] * 0.98 or last["match"] > first["match"] + 0.01
    summary = {
        "ok": bool(improved and skipped < args.steps // 2),
        "stage": args.stage,
        "teacher": teacher_id,
        "n_seq": len(seqs),
        "seq_len": args.seq_len,
        "steps": args.steps,
        "beta": args.beta,
        "draft_params": count_params(student),
        "first": first,
        "last": last,
        "skipped": skipped,
        "out": str(out_dir),
        "seconds": round(time.time() - t0, 1),
    }
    Path(kaggle_out("distill_summary.json")).write_text(json.dumps(summary, indent=2))
    (out_dir / "train_history.json").write_text(json.dumps(history))
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["ok"]:
        print("WARN: loss/match did not clearly improve; still wrote checkpoint", flush=True)


if __name__ == "__main__":
    main()
