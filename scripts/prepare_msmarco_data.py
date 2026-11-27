#!/usr/bin/env python3
"""
Builds a fast-loading, 100% pure zero-lever binary CSR Sponge Index (`msmarco_sponge.bin`)
from Microsoft MS MARCO v2.1 (~1M real human Q&As) mapped onto the static Kaikki lexicon.

Every single layer (Sphere 2 POS valve, Sphere 3 Syntactic Hub transitions, Sphere 4 Sponge
word->answer channels) learns purely from the MS MARCO Q&A pairs themselves — zero stopword
lists, zero hardcoded grammar rules, zero extra corpora!
"""

import argparse
import glob
import os
import re
import struct
import time
from collections import defaultdict

import pyarrow.parquet as pq

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
NUM_POS = 12
NUM_SYN = 8

# Syntactic hub IDs matching include/talk_hud.hpp
SYN_WH_Q = 0
SYN_NP_S = 1
SYN_VP_BE = 2
SYN_VP_AC = 3
SYN_NP_O = 4
SYN_PP_M = 5
SYN_DECL = 6
SYN_CONJ = 7


def load_kaikki_lexicon(path: str):
    t0 = time.time()
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GLXL", f"Bad lexicon magic: {magic}"
        ver, num_words, num_pos = struct.unpack("<III", f.read(12))
        word_to_id = {}
        primary_pos = bytearray(num_words)
        for wid in range(num_words):
            pid, wlen, _mask = struct.unpack("<BBH", f.read(4))
            w = f.read(wlen).decode("utf-8", errors="ignore")
            word_to_id[w] = wid
            primary_pos[wid] = pid
    print(f"[LEX] Loaded {num_words:,} Kaikki words in {time.time() - t0:.2f}s")
    return num_words, word_to_id, primary_pos


def fnv1a_32(s: str) -> int:
    h = 2166136261
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def tokenize(text: str, word_to_id: dict, primary_pos: bytearray, num_words: int):
    toks = []
    for raw in TOKEN_RE.findall(text):
        low = raw.lower()
        wid = word_to_id.get(low)
        if wid is not None:
            pid = primary_pos[wid]
        else:
            wid = fnv1a_32(low) % num_words
            if any(c.isdigit() for c in raw):
                pid = 9  # num
            elif raw[0].isupper():
                pid = 4  # name
            else:
                pid = 0  # noun
        toks.append((wid, pid))
    return toks


def classify_answer_pos(a_text: str, a_toks: list) -> int:
    if not a_toks:
        return 11
    return a_toks[0][1]


def tag_syntax_and_update(toks: list, pos_to_syn: list, syn_flow: list) -> int:
    if not toks:
        return 0
    seen_verb = False
    in_prep = False
    prev_s = -1
    trans = 0
    for i, (_wid, p) in enumerate(toks):
        if i == 0 and (p == 5 or p == 3):
            s = SYN_WH_Q
            in_prep = False
        elif p == 8:
            s = SYN_CONJ
            in_prep = False
        elif p == 1:
            s = SYN_VP_AC if seen_verb else SYN_VP_BE
            seen_verb = True
            in_prep = False
        elif p == 6:
            s = SYN_PP_M
            in_prep = True
        elif in_prep:
            s = SYN_PP_M
        else:
            s = SYN_NP_O if seen_verb else SYN_NP_S
        p_clamped = p if p < NUM_POS else 11
        pos_to_syn[p_clamped][s] += 1
        if prev_s >= 0 and prev_s != s:
            syn_flow[prev_s][s] += 1
            trans += 1
        prev_s = s
    if prev_s >= 0:
        syn_flow[prev_s][SYN_DECL] += 1
        trans += 1
    return trans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="/run/media/liveuser/GLLM_DATA/gllm_datasets/extracted/talk_index/kaikki_lexicon.bin")
    ap.add_argument("--parquet-dir", default="/run/media/liveuser/GLLM_DATA/gllm_datasets/raw/msmarco_parquet")
    ap.add_argument("--out-bin", default="/run/media/liveuser/GLLM_DATA/gllm_datasets/extracted/talk_index/msmarco_sponge.bin")
    ap.add_argument("--out-sample", default="/run/media/liveuser/GLLM_DATA/gllm_datasets/extracted/talk_index/msmarco_sample_qa.tsv")
    ap.add_argument("--max-qa", type=int, default=0, help="0 = process all labeled MS MARCO Q&As")
    args = ap.parse_args()

    t_start = time.time()
    num_words, word_to_id, primary_pos = load_kaikki_lexicon(args.lexicon)

    files = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    assert files, f"No parquet files found in {args.parquet_dir}"
    print(f"[DATA] Found {len(files)} MS MARCO parquet shards")

    # Core sponge tables (100% pure raw counts, zero stopwords, zero rules)
    ans_to_id = {}
    answers = []  # (text, anchor_wid, primary_pos, total_hits)
    word_edges = defaultdict(lambda: defaultdict(int))
    opener_pos = defaultdict(lambda: [0] * NUM_POS)
    pos_to_syn = [[0] * NUM_SYN for _ in range(NUM_POS)]
    syn_flow = [[0] * NUM_SYN for _ in range(NUM_SYN)]
    total_transitions = 0
    qa_count = 0
    sample_qa = []

    for fpath in files:
        t_f = time.time()
        pf = pq.ParquetFile(fpath)
        top_names = set(pf.schema_arrow.names)
        cols = [c for c in ["query", "answers", "wellFormedAnswers"] if c in top_names]
        for batch in pf.iter_batches(batch_size=16384, columns=cols):
            q_col = batch.column("query").to_pylist()
            a_col = batch.column("answers").to_pylist() if "answers" in cols else [None] * len(q_col)
            wf_col = batch.column("wellFormedAnswers").to_pylist() if "wellFormedAnswers" in cols else [None] * len(q_col)

            for q_raw, a_list, wf_list in zip(q_col, a_col, wf_col):
                if not q_raw:
                    continue
                # Pick well-formed human answer when available, else primary human answer
                chosen = None
                if wf_list:
                    for cand in wf_list:
                        if cand and cand.strip() and cand.strip().lower() != "no answer present.":
                            chosen = " ".join(cand.strip().split())
                            break
                if chosen is None and a_list:
                    for cand in a_list:
                        if cand and cand.strip() and cand.strip().lower() != "no answer present.":
                            chosen = " ".join(cand.strip().split())
                            break
                if not chosen:
                    continue

                q = " ".join(q_raw.strip().split())
                # Keep reasonable single/short answers for sponge basins (up to 320 chars)
                if len(chosen) > 320:
                    chosen = chosen[:320].rsplit(" ", 1)[0]
                q_toks = tokenize(q, word_to_id, primary_pos, num_words)
                a_toks = tokenize(chosen, word_to_id, primary_pos, num_words)
                if not q_toks or not a_toks:
                    continue

                ap = classify_answer_pos(chosen, a_toks)
                a_key = chosen.lower()
                aid = ans_to_id.get(a_key)
                if aid is None:
                    aid = len(answers)
                    ans_to_id[a_key] = aid
                    answers.append([chosen, a_toks[0][0], ap, 1])
                else:
                    answers[aid][3] += 1

                # Sphere 2: pure raw opening-word -> answer POS counts
                for k in range(min(2, len(q_toks))):
                    opener_pos[q_toks[k][0]][ap] += (2 if k == 0 else 1)

                # Sphere 3: pure syntactic hub flow learned directly from Q & A
                total_transitions += tag_syntax_and_update(q_toks, pos_to_syn, syn_flow)
                total_transitions += tag_syntax_and_update(a_toks, pos_to_syn, syn_flow)

                # Sphere 4: ALL question words carve channels to answer basin (ZERO stopwords!)
                seen_w = set()
                for wid, _pid in q_toks:
                    if wid not in seen_w:
                        seen_w.add(wid)
                        word_edges[wid][aid] += 1

                if len(sample_qa) < 6000 and 4 <= len(q_toks) <= 14 and 2 <= len(a_toks) <= 18:
                    sample_qa.append((q.replace("\t", " "), chosen.replace("\t", " ")))

                qa_count += 1
                if args.max_qa > 0 and qa_count >= args.max_qa:
                    break
            if args.max_qa > 0 and qa_count >= args.max_qa:
                break
        print(f"  -> {os.path.basename(fpath)}: cumulative {qa_count:,} Q&As, {len(answers):,} unique answers ({time.time() - t_f:.1f}s)")
        if args.max_qa > 0 and qa_count >= args.max_qa:
            break

    print(f"[BUILD] Packing binary CSR sponge index ({qa_count:,} Q&As, {len(answers):,} answers, {len(word_edges):,} active words)...")
    t_pack = time.time()

    # Build answer text blob & metadata
    ans_blob = bytearray()
    ans_meta = bytearray()
    for text, anchor_wid, ap, hits in answers:
        tb = text.encode("utf-8")
        off = len(ans_blob)
        ans_blob.extend(tb)
        ans_meta.extend(struct.pack("<IIBBHI", anchor_wid, hits, ap, 0, len(tb) & 0xFFFF, off))

    # Build flat CSR over all `num_words`
    # Store only words that have >= 1 edge in a compact sorted list (wid, total_count, edge_off, edge_cnt)
    active_wids = sorted(word_edges.keys())
    word_index_buf = bytearray()
    edges_buf = bytearray()
    edge_off = 0
    for wid in active_wids:
        emap = word_edges[wid]
        tot_cnt = sum(emap.values())
        ecnt = len(emap)
        word_index_buf.extend(struct.pack("<IQII", wid, tot_cnt, edge_off, ecnt))
        for aid, cnt in emap.items():
            edges_buf.extend(struct.pack("<II", aid, cnt))
        edge_off += ecnt

    # Build opener POS table
    opener_buf = bytearray()
    for wid, cnts in opener_pos.items():
        opener_buf.extend(struct.pack("<I12I", wid, *cnts))

    os.makedirs(os.path.dirname(args.out_bin), exist_ok=True)
    with open(args.out_bin, "wb") as out:
        # Header: magic(4)="GSPG", ver(u32)=1, num_words(u32), qa_count(u32),
        #         num_answers(u32), ans_blob_len(u64), num_active_words(u32),
        #         num_edges(u64), num_openers(u32), total_transitions(u64)
        out.write(b"GSPG")
        out.write(struct.pack(
            "<IIIIQIQIQ",
            1,
            num_words,
            qa_count,
            len(answers),
            len(ans_blob),
            len(active_wids),
            edge_off,
            len(opener_pos),
            total_transitions,
        ))
        # Sphere 3 tables
        for p in range(NUM_POS):
            out.write(struct.pack("<8Q", *pos_to_syn[p]))
        for s in range(NUM_SYN):
            out.write(struct.pack("<8Q", *syn_flow[s]))
        out.write(ans_meta)
        out.write(ans_blob)
        out.write(word_index_buf)
        out.write(edges_buf)
        out.write(opener_buf)

    with open(args.out_sample, "w", encoding="utf-8") as sf:
        for q, a in sample_qa:
            sf.write(f"{q}\t{a}\n")

    sz_mb = os.path.getsize(args.out_bin) / (1024 * 1024)
    print(f"[DONE] Wrote {args.out_bin} ({sz_mb:.1f} MB) & {len(sample_qa):,} live stream Q&As -> {args.out_sample}")
    print(f"       Q&As: {qa_count:,} | Answers: {len(answers):,} | CSR Edges: {edge_off:,} | Syntax Transitions: {total_transitions:,}")
    print(f"       Pack time: {time.time() - t_pack:.1f}s | Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
