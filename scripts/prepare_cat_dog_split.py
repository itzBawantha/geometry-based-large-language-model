#!/usr/bin/env python3
"""
Prepare deterministic 80/20 stratified train/test split manifest (TSV)
for all complete 6-landmark CatFLW and DogFLW RGB images from
data/mammal_landmarks/annotations.jsonl.

Strictly uses exact dataset face bounding boxes and exact feature polygon
bounding boxes (ear_right, ear_left, eye_right, eye_left, nose, mouth),
cleanly filtering any NaN polygon entries.
"""

import json
import math
import os
import random

ANNOT_PATH = "data/mammal_landmarks/annotations.jsonl"
OUT_TSV    = "data/cat_dog_split.tsv"
FEATURES   = ("ear_right", "ear_left", "eye_right", "eye_left", "nose", "mouth")


def poly_bbox(pts):
    if not pts:
        return None
    xs, ys = [], []
    for p in pts:
        if len(p) < 2:
            return None
        x, y = float(p[0]), float(p[1])
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            return None
        xs.append(x)
        ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def main():
    by_species = {"cat": [], "dog": []}
    skipped = 0

    with open(ANNOT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src = r.get("source", "")
            if src not in ("catflw", "dogflw"):
                continue
            sp = "cat" if src == "catflw" else "dog"
            img_path = r.get("image_path", "")
            if not img_path or not os.path.isfile(img_path):
                skipped += 1
                continue
            bbox = r.get("bbox")
            if not bbox or len(bbox) != 4 or any(math.isnan(float(v)) for v in bbox):
                skipped += 1
                continue
            polys = r.get("polygons", {})
            feat_boxes = []
            ok = True
            for fn in FEATURES:
                fb = poly_bbox(polys.get(fn))
                if fb is None:
                    ok = False
                    break
                feat_boxes.append(fb)
            if not ok:
                skipped += 1
                continue
            by_species[sp].append((img_path, tuple(float(v) for v in bbox), feat_boxes))

    rng = random.Random(42)
    rows = []
    counts = {}
    for sp in ("cat", "dog"):
        items = by_species[sp]
        items.sort(key=lambda x: x[0])  # deterministic base order before shuffle
        rng.shuffle(items)
        n_test = int(round(len(items) * 0.20))
        test_set  = items[:n_test]
        train_set = items[n_test:]
        counts[sp] = (len(train_set), len(test_set))
        for split_name, subset in (("train", train_set), ("test", test_set)):
            for img_path, fbox, feat_boxes in subset:
                cols = [
                    split_name,
                    sp,
                    img_path,
                    f"{fbox[0]:.2f},{fbox[1]:.2f},{fbox[2]:.2f},{fbox[3]:.2f}",
                ]
                for fb in feat_boxes:
                    cols.append(f"{fb[0]:.2f},{fb[1]:.2f},{fb[2]:.2f},{fb[3]:.2f}")
                rows.append("\t".join(cols))

    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    with open(OUT_TSV, "w", encoding="utf-8") as out:
        for line in rows:
            out.write(line + "\n")

    print(f"[SPLIT] Wrote {OUT_TSV}: {len(rows)} valid samples (skipped {skipped} NaN/invalid)")
    for sp in ("cat", "dog"):
        tr, te = counts[sp]
        print(f"        {sp:4s}: train={tr:4d} | test={te:4d} | total={tr + te:4d}")
    tot_tr = counts["cat"][0] + counts["dog"][0]
    tot_te = counts["cat"][1] + counts["dog"][1]
    print(f"        ALL : train={tot_tr:4d} | test={tot_te:4d} | total={tot_tr + tot_te:4d}")


if __name__ == "__main__":
    main()
