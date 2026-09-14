"""Mixed corpora for Base packing and Instruct prompt dumps. Each source is optional."""

from __future__ import annotations

import random
from typing import Callable


def _ok(text: str, min_len: int = 40) -> bool:
    t = (text or "").strip()
    return len(t) >= min_len


def _try(name: str, fn: Callable[[], list[str]]) -> list[str]:
    try:
        rows = fn()
        print(f"mix {name}: {len(rows)} texts", flush=True)
        return rows
    except Exception as e:
        print(f"mix skip {name}: {type(e).__name__}: {e}", flush=True)
        return []


def _take_split(ds, field: str, n: int, extra: str | None = None) -> list[str]:
    out = []
    for row in ds:
        parts = [str(row.get(field) or "")]
        if extra:
            parts.append(str(row.get(extra) or ""))
        text = "\n".join(p for p in parts if p)
        if _ok(text):
            out.append(text.strip())
        if len(out) >= n:
            break
    return out


def load_base_texts(per_source: int) -> dict[str, list[str]]:
    from datasets import load_dataset

    sources: dict[str, list[str]] = {}

    def wiki():
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        return _take_split(ds, "text", per_source * 4)

    def wiki_en():
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        except Exception:
            ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        out = []
        for row in ds:
            t = (row.get("text") or "").strip()
            if _ok(t, 200):
                out.append(t[:4000])
            if len(out) >= per_source:
                break
        return out

    def news():
        ds = load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True)
        out = []
        for row in ds:
            t = (row.get("article") or "").strip()
            if _ok(t, 80):
                out.append(t[:4000])
            if len(out) >= per_source:
                break
        return out

    def math():
        out = []
        gsm = load_dataset("gsm8k", "main", split="train")
        for row in gsm:
            out.append(f"Question: {row['question']}\nAnswer: {row['answer']}")
            if len(out) >= per_source // 2:
                break
        try:
            mh = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
            for row in mh:
                out.append(f"{row.get('problem','')}\n{row.get('solution','')}".strip())
                if len(out) >= per_source:
                    break
        except Exception as e:
            print("hendrycks_math skip", e, flush=True)
        return out[:per_source]

    def code():
        ds = load_dataset("code_search_net", "python", split="train", streaming=True, trust_remote_code=True)
        out = []
        for row in ds:
            t = (row.get("func_code_string") or row.get("whole_func_string") or "").strip()
            if _ok(t, 80):
                out.append(t[:4000])
            if len(out) >= per_source:
                break
        return out

    def science():
        ds = load_dataset("scientific_papers", "arxiv", split="train", streaming=True, trust_remote_code=True)
        out = []
        for row in ds:
            t = ((row.get("abstract") or "") + "\n" + (row.get("article") or "")).strip()
            if _ok(t, 80):
                out.append(t[:4000])
            if len(out) >= per_source:
                break
        return out

    for name, fn in (
        ("wikitext103", wiki),
        ("wikipedia", wiki_en),
        ("cnn_dailymail", news),
        ("math", math),
        ("python_code", code),
        ("arxiv", science),
    ):
        rows = _try(name, fn)
        if rows:
            sources[name] = rows
    return sources


def load_instruct_prompts(per_source: int) -> dict[str, list[str]]:
    from datasets import load_dataset

    sources: dict[str, list[str]] = {}

    def gsm():
        ds = load_dataset("gsm8k", "main", split="train")
        return [row["question"] for row in ds][:per_source]

    def math():
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
        return [row["problem"] for row in ds if row.get("problem")][:per_source]

    def alpaca():
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        out = []
        for row in ds:
            inst = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            q = inst if not inp else f"{inst}\n{inp}"
            if _ok(q, 20):
                out.append(q)
            if len(out) >= per_source:
                break
        return out

    def dolly():
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        out = []
        for row in ds:
            inst = (row.get("instruction") or "").strip()
            ctx = (row.get("context") or "").strip()
            q = inst if not ctx else f"{inst}\n\n{ctx}"
            if _ok(q, 20):
                out.append(q)
            if len(out) >= per_source:
                break
        return out

    def mbpp():
        ds = load_dataset("mbpp", split="train")
        return [row["text"] for row in ds if row.get("text")][:per_source]

    def codealpaca():
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        out = []
        for row in ds:
            inst = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            q = inst if not inp else f"{inst}\n{inp}"
            if _ok(q, 20):
                out.append(q)
            if len(out) >= per_source:
                break
        return out

    for name, fn in (
        ("gsm8k", gsm),
        ("hendrycks_math", math),
        ("alpaca", alpaca),
        ("dolly", dolly),
        ("mbpp", mbpp),
        ("codealpaca", codealpaca),
    ):
        rows = _try(name, fn)
        if rows:
            sources[name] = rows
    return sources


def pack_texts(tok, texts: list[str], max_len: int, n: int) -> list[list[int]]:
    buf: list[int] = []
    seqs: list[list[int]] = []
    for text in texts:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids + [tok.eos_token_id or 1])
        while len(buf) >= max_len:
            seqs.append(buf[:max_len])
            buf = buf[max_len:]
            if len(seqs) >= n:
                return seqs
    return seqs[:n]


def interleave_packed(tok, sources: dict[str, list[str]], max_len: int, n: int, seed: int = 0) -> tuple[list[list[int]], dict]:
    rng = random.Random(seed)
    per = max(32, n // max(len(sources), 1))
    packed: dict[str, list[list[int]]] = {}
    for name, texts in sources.items():
        rng.shuffle(texts)
        packed[name] = pack_texts(tok, texts, max_len, per)
        print(f"packed {name}: {len(packed[name])} seqs", flush=True)
    names = [k for k, v in packed.items() if v]
    seqs: list[list[int]] = []
    idx = {k: 0 for k in names}
    while len(seqs) < n and names:
        progressed = False
        for name in list(names):
            i = idx[name]
            if i >= len(packed[name]):
                names.remove(name)
                continue
            seqs.append(packed[name][i])
            idx[name] = i + 1
            progressed = True
            if len(seqs) >= n:
                break
        if not progressed:
            break
    rng.shuffle(seqs)
    counts = {k: idx.get(k, 0) for k in packed}
    return seqs[:n], counts


def round_robin_prompts(sources: dict[str, list[str]], n: int, seed: int = 0) -> tuple[list[str], dict]:
    rng = random.Random(seed)
    bags = {k: list(v) for k, v in sources.items() if v}
    for v in bags.values():
        rng.shuffle(v)
    names = list(bags)
    out: list[str] = []
    idx = {k: 0 for k in names}
    counts: dict[str, int] = {k: 0 for k in names}
    while len(out) < n and names:
        for name in list(names):
            i = idx[name]
            if i >= len(bags[name]):
                names.remove(name)
                continue
            out.append(bags[name][i])
            idx[name] = i + 1
            counts[name] += 1
            if len(out) >= n:
                break
    return out[:n], counts
