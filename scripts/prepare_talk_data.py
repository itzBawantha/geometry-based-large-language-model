#!/usr/bin/env python3
"""
Builds compact, fast-loading binary lexicon + English Q&A index on GLLM_DATA
from Kaikki English Wiktionary JSONL and SQuAD v1.1 for real-time CUDA visualization.

Outputs (in --out-dir):
  - kaikki_lexicon.bin : All unique Kaikki English words with primary POS & POS bitmask
  - squad_qa.tsv       : Clean English Question \t Answer pairs ready for real-time streaming
"""

import argparse
import json
import os
import re
import struct
import time
from collections import Counter

POS_MAP = {
    "noun": 0,
    "verb": 1,
    "adj": 2,
    "adv": 3,
    "name": 4,
    "pron": 5,
    "prep": 6,
    "prep_phrase": 6,
    "det": 7,
    "article": 7,
    "conj": 8,
    "num": 9,
    "intj": 10,
    "particle": 10,
    "contraction": 10,
}
POS_NAMES = [
    "noun", "verb", "adj", "adv", "name", "pron",
    "prep", "det", "conj", "num", "intj", "other",
]
# Priority when a word has multiple POS (prefers core grammatical classes over rare/other)
POS_PRIORITY = {
    5: 0,   # pron
    7: 1,   # det/article
    6: 2,   # prep
    8: 3,   # conj
    1: 4,   # verb
    0: 5,   # noun
    2: 6,   # adj
    3: 7,   # adv
    9: 8,   # num
    4: 9,   # name
    10: 10, # intj/particle
    11: 11, # other
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def build_lexicon(kaikki_path: str, out_bin: str) -> dict:
    t0 = time.time()
    print(f"[LEX] Reading Kaikki JSONL: {kaikki_path}")
    # Map lowercased word -> [pos_mask, primary_pos]
    # We keep ALL unique lowercased words from Kaikki (~1.3M+) so the outer sphere
    # literally contains the full English dictionary!
    words = {}
    total_lines = 0
    with open(kaikki_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            w = obj.get("word")
            p = obj.get("pos")
            if not w or not p:
                continue
            wl = w.strip().lower()
            if not wl or len(wl) > 48:
                continue
            pid = POS_MAP.get(p, 11)
            bit = 1 << pid
            cur = words.get(wl)
            if cur is None:
                words[wl] = [bit, pid]
            else:
                cur[0] |= bit
                if POS_PRIORITY[pid] < POS_PRIORITY[cur[1]]:
                    cur[1] = pid

    # Ensure core English function words have crisp canonical primary POS
    canonical_overrides = {
        "what": 5, "who": 5, "whom": 5, "whose": 5, "which": 5, "where": 3, "when": 3, "why": 3, "how": 3,
        "i": 5, "you": 5, "he": 5, "she": 5, "it": 5, "we": 5, "they": 5, "me": 5, "him": 5, "her": 5, "us": 5, "them": 5,
        "the": 7, "a": 7, "an": 7, "this": 7, "that": 7, "these": 7, "those": 7, "my": 7, "your": 7, "his": 7, "its": 7, "our": 7, "their": 7,
        "is": 1, "are": 1, "was": 1, "were": 1, "be": 1, "been": 1, "being": 1, "am": 1,
        "do": 1, "does": 1, "did": 1, "have": 1, "has": 1, "had": 1, "can": 1, "could": 1, "will": 1, "would": 1, "should": 1, "may": 1, "might": 1,
        "in": 6, "on": 6, "at": 6, "to": 6, "for": 6, "of": 6, "from": 6, "with": 6, "by": 6, "about": 6, "into": 6, "through": 6, "between": 6, "during": 6, "under": 6, "over": 6, "after": 6, "before": 6,
        "and": 8, "or": 8, "but": 8, "because": 8, "if": 8, "while": 8, "although": 8, "than": 8, "so": 8,
        "not": 3, "no": 7, "yes": 10,
    }
    for cw, cpid in canonical_overrides.items():
        if cw in words:
            words[cw][0] |= (1 << cpid)
            words[cw][1] = cpid
        else:
            words[cw] = [1 << cpid, cpid]

    # Sort deterministically by (primary_pos, word) so outer-sphere golden-spiral
    # placement naturally groups or cleanly distributes every word with stable IDs!
    sorted_items = sorted(words.items(), key=lambda kv: (kv[1][1], kv[0]))
    n_words = len(sorted_items)

    # Compact binary format "GLXL" v1:
    #   magic[4] = "GLXL", version(u32)=1, num_words(u32), num_pos(u32)=12
    #   For each word: primary_pos(u8), word_len(u8), pos_mask(u16), word_bytes[word_len]
    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    pos_counts = Counter()
    with open(out_bin, "wb") as out:
        out.write(b"GLXL")
        out.write(struct.pack("<III", 1, n_words, len(POS_NAMES)))
        for w, (mask, pid) in sorted_items:
            wb = w.encode("utf-8")
            if len(wb) > 250:
                wb = wb[:250]
            pos_counts[pid] += 1
            out.write(struct.pack("<BBH", pid, len(wb), mask & 0xFFFF))
            out.write(wb)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(out_bin) / (1024 * 1024)
    print(f"[LEX] Wrote {n_words:,} unique words from {total_lines:,} Kaikki entries -> {out_bin} ({size_mb:.1f} MB) in {elapsed:.1f}s")
    for pid, name in enumerate(POS_NAMES):
        print(f"      POS #{pid:2d} {name:6s}: {pos_counts[pid]:,}")
    return words


def build_squad_qa(squad_files: list, out_tsv: str) -> int:
    t0 = time.time()
    pairs = []
    seen_q = set()
    for path in squad_files:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for art in data.get("data", []):
            title = art.get("title", "").replace("_", " ")
            for para in art.get("paragraphs", []):
                for qa in para.get("qas", []):
                    q = " ".join(qa.get("question", "").strip().split())
                    ans_list = qa.get("answers", [])
                    if not q or not ans_list:
                        continue
                    a = " ".join(ans_list[0].get("text", "").strip().split())
                    if not a:
                        continue
                    q_toks = TOKEN_RE.findall(q)
                    a_toks = TOKEN_RE.findall(a)
                    if not (3 <= len(q_toks) <= 18 and 1 <= len(a_toks) <= 8):
                        continue
                    ql = q.lower()
                    if ql in seen_q:
                        continue
                    seen_q.add(ql)
                    # Store (starter_word, len_q, q, a, title)
                    starter = q_toks[0].lower()
                    pairs.append((starter, len(q_toks), q, a, title))

    # Interleave crisp short questions from shared WH-stems ("what", "who", "when", "where", "how", "which")
    # early in the stream so the user immediately sees water-droplet merging on shared stems AND splitting
    # into distinct answer branches!
    wh_order = ["what", "who", "where", "when", "which", "how", "why"]
    buckets = {w: [] for w in wh_order}
    other = []
    for p in pairs:
        if p[0] in buckets:
            buckets[p[0]].append(p)
        else:
            other.append(p)
    for k in buckets:
        buckets[k].sort(key=lambda x: (x[1], x[2]))
    other.sort(key=lambda x: (x[1], x[2]))

    # Build a rich opening sequence: blocks of 6 similar-stem questions so merging is unmistakable,
    # followed by round-robin across all remaining SQuAD Q&A pairs!
    ordered = []
    idx_map = {w: 0 for w in wh_order}
    # First ~2400 pairs in mini-bursts of 5 per WH-stem so shared arteries glow immediately
    for _ in range(70):
        for w in wh_order:
            b = buckets[w]
            s = idx_map[w]
            chunk = b[s:s + 5]
            ordered.extend(chunk)
            idx_map[w] = s + len(chunk)
    # Remainder round-robin
    remaining = True
    while remaining:
        remaining = False
        for w in wh_order:
            b = buckets[w]
            if idx_map[w] < len(b):
                ordered.append(b[idx_map[w]])
                idx_map[w] += 1
                remaining = True
    ordered.extend(other)

    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8") as out:
        for _, _, q, a, title in ordered:
            q_clean = q.replace("\t", " ").replace("\n", " ")
            a_clean = a.replace("\t", " ").replace("\n", " ")
            out.write(f"{q_clean}\t{a_clean}\n")

    print(f"[QA]  Wrote {len(ordered):,} clean English Q&A pairs -> {out_tsv} in {time.time() - t0:.1f}s")
    return len(ordered)


def build_squad_sentences(squad_files: list, out_tsv: str) -> int:
    t0 = time.time()
    sent_re = re.compile(r"(?<=[.!?])\s+")
    sents = []
    seen = set()
    for path in squad_files:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for art in data.get("data", []):
            for para in art.get("paragraphs", []):
                ctx = para.get("context", "")
                for s in sent_re.split(ctx):
                    s = " ".join(s.strip().split())
                    if 5 <= len(TOKEN_RE.findall(s)) <= 24 and s not in seen:
                        seen.add(s)
                        sents.append(s)
    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8") as out:
        for s in sents:
            out.write(s + "\n")
    print(f"[SYN] Wrote {len(sents):,} real English sentences -> {out_tsv} in {time.time() - t0:.1f}s")
    return len(sents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaikki", default="/run/media/liveuser/GLLM_DATA/gllm_datasets/extracted/kaikki_english/kaikki.org-dictionary-English.jsonl")
    ap.add_argument("--squad-train", default="/run/media/liveuser/GLLM_DATA/gllm_datasets/raw/squad-train-v1.1.json")
    ap.add_argument("--squad-dev", default="/run/media/liveuser/GLLM_DATA/gllm_datasets/raw/squad-dev-v1.1.json")
    ap.add_argument("--out-dir", default="/run/media/liveuser/GLLM_DATA/gllm_datasets/extracted/talk_index")
    args = ap.parse_args()

    build_lexicon(args.kaikki, os.path.join(args.out_dir, "kaikki_lexicon.bin"))
    build_squad_qa([args.squad_train, args.squad_dev], os.path.join(args.out_dir, "squad_qa.tsv"))
    build_squad_sentences([args.squad_train, args.squad_dev], os.path.join(args.out_dir, "squad_sentences.tsv"))


if __name__ == "__main__":
    main()
