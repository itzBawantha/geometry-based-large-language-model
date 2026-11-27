#!/usr/bin/env python3
"""
Streams OpenHermes-2.5 (1M curated GPT-4 Q&As) JSON into clean TSV pairs for
`bin/gllm_pack_sponge`, escaping newlines as \\n so multi-step in-depth answers
preserve formatting in the scrollable 4-Sphere HUD.
"""

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", default="/run/media/liveuser/GLLM_DATA/gllm_datasets/raw/openhermes/openhermes2_5.json")
    ap.add_argument("--out-tsv", default="/run/media/liveuser/GLLM_DATA/gllm_datasets/extracted/talk_index/openhermes_qa.tsv")
    ap.add_argument("--out-sample", default="/run/media/liveuser/GLLM_DATA/gllm_datasets/extracted/talk_index/openhermes_sample_qa.tsv")
    ap.add_argument("--max-ans-chars", type=int, default=2000)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[OH2.5] Loading {args.in_json}...", flush=True)
    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[OH2.5] Parsed {len(data):,} conversations in {time.time() - t0:.1f}s. Extracting Q&A turns...", flush=True)

    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)
    n_pairs = 0
    sample = []

    with open(args.out_tsv, "w", encoding="utf-8") as out:
        for item in data:
            convs = item.get("conversations", [])
            pending_q = None
            for turn in convs:
                role = turn.get("from", "")
                val = turn.get("value", "")
                if not val or not isinstance(val, str):
                    continue
                if role in ("human", "user"):
                    pending_q = " ".join(val.strip().split())
                elif role in ("gpt", "assistant") and pending_q:
                    a = val.strip()
                    if not a:
                        pending_q = None
                        continue
                    # Cap extreme multi-page essays cleanly at sentence/word boundary
                    if len(a) > args.max_ans_chars:
                        a = a[:args.max_ans_chars].rsplit(" ", 1)[0] + "..."
                    q_clean = pending_q.replace("\t", " ").replace("\r", " ").replace("\n", " ")
                    # Escape real newlines inside rich answers as literal \n for TSV transport
                    a_clean = a.replace("\t", " ").replace("\r", "").replace("\\", "\\\\").replace("\n", "\\n")
                    if q_clean and a_clean:
                        out.write(f"{q_clean}\t{a_clean}\n")
                        n_pairs += 1
                        if len(sample) < 6000 and 12 <= len(q_clean) <= 100 and 20 <= len(a) <= 220:
                            sample.append((q_clean, " ".join(a.split()).replace("\t", " ")))
                    pending_q = None

    with open(args.out_sample, "w", encoding="utf-8") as sf:
        for q, a in sample:
            sf.write(f"{q}\t{a}\n")

    mb = os.path.getsize(args.out_tsv) / (1024 * 1024)
    print(f"[OH2.5] Wrote {n_pairs:,} rich GPT-4 Q&A pairs -> {args.out_tsv} ({mb:.1f} MB) in {time.time() - t0:.1f}s", flush=True)
    print(f"[OH2.5] Wrote {len(sample):,} live stream Q&As -> {args.out_sample}", flush=True)


if __name__ == "__main__":
    main()
