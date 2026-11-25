#!/usr/bin/env python3
"""
Prepare any custom image folder dataset for GLLM (bin/gllm_train --folder).

Expected input folder layout:
    <input_dir>/
        <class_A>/
            img001.png
            img002.jpg
        <class_B>/
            ...

Outputs clean binary PGM (P5) grayscale images of uniform W x H size ready for:
    ./bin/gllm_train --model models/custom.gllm --folder <output_dir>

Supports:
  - PNG / JPG / JPEG / BMP / WEBP / TIF / PGM / PPM (via Pillow, ImageMagick, or ffmpeg)
  - --auto-invert : detects dark ink on light paper and inverts to bright-on-black
  - --invert      : unconditionally invert intensity (255 - px)
  - --center      : tight-bbox aspect-preserving centering (matches GLLM 22/28 frame)
  - --generate-demo <out_dir> : generates a zero-dependency 3-class geometric demo
                                (circle, square, triangle) to test custom training immediately
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".pgm", ".ppm"}


def write_pgm(path: str, w: int, h: int, pixels: list[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(bytes(max(0, min(255, int(v))) for v in pixels))


def read_pgm_p5(path: str) -> tuple[int, int, list[int]]:
    with open(path, "rb") as f:
        data = f.read()
    idx = 0
    tokens = []
    while len(tokens) < 4:
        while idx < len(data) and data[idx] in b" \t\r\n":
            idx += 1
        if idx < len(data) and data[idx] == ord("#"):
            while idx < len(data) and data[idx] != ord("\n"):
                idx += 1
            continue
        start = idx
        while idx < len(data) and data[idx] not in b" \t\r\n":
            idx += 1
        tokens.append(data[start:idx].decode("ascii"))
    if tokens[0] != "P5":
        raise ValueError(f"Expected P5 PGM, got {tokens[0]}")
    w, h, maxv = int(tokens[1]), int(tokens[2]), int(tokens[3])
    idx += 1  # single whitespace byte after maxv
    raw = data[idx : idx + w * h]
    if maxv != 255 and maxv > 0:
        px = [int(b * 255 / maxv) for b in raw]
    else:
        px = list(raw)
    return w, h, px


def load_grayscale_resized(src_path: str, size: int) -> list[int]:
    # 1. Try Pillow (fastest, purest)
    try:
        from PIL import Image  # type: ignore

        with Image.open(src_path) as im:
            im = im.convert("L").resize((size, size), Image.Resampling.BILINEAR)
            return list(im.getdata())
    except ImportError:
        pass

    # 2. Fallback: ImageMagick (magick / convert)
    for cmd in ("magick", "convert"):
        if shutil.which(cmd):
            with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as tf:
                tmp_pgm = tf.name
            try:
                subprocess.run(
                    [cmd, src_path, "-colorspace", "Gray", "-resize", f"{size}x{size}!", "-depth", "8", tmp_pgm],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                _, _, px = read_pgm_p5(tmp_pgm)
                return px
            finally:
                if os.path.exists(tmp_pgm):
                    os.unlink(tmp_pgm)

    # 3. Fallback: ffmpeg
    if shutil.which("ffmpeg"):
        with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as tf:
            tmp_pgm = tf.name
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", src_path,
                    "-vf", f"format=gray,scale={size}:{size}",
                    "-frames:v", "1", tmp_pgm,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _, _, px = read_pgm_p5(tmp_pgm)
            return px
        finally:
            if os.path.exists(tmp_pgm):
                os.unlink(tmp_pgm)

    # 4. Native P5 PGM direct pass-through if already matching size
    if src_path.lower().endswith(".pgm"):
        w, h, px = read_pgm_p5(src_path)
        if w == size and h == size:
            return px

    raise RuntimeError(
        "No image decoder found for non-PGM images. Install Pillow (`pip install pillow`), "
        "ImageMagick (`magick`/`convert`), or `ffmpeg`."
    )


def maybe_invert(px: list[int], force_invert: bool, auto_invert: bool) -> list[int]:
    if force_invert:
        return [255 - v for v in px]
    if auto_invert:
        # Sample border pixels: if border average is bright (>128), it's dark ink on light paper
        n = int(round(math.sqrt(len(px))))
        border = []
        for x in range(n):
            border.append(px[x])
            border.append(px[(n - 1) * n + x])
        for y in range(n):
            border.append(px[y * n])
            border.append(px[y * n + (n - 1)])
        if sum(border) / max(1, len(border)) > 128.0:
            return [255 - v for v in px]
    return px


def center_canonical(px: list[int], size: int) -> list[int]:
    mx = max(px) if px else 0
    if mx < 8:
        return px
    thr = mx * 0.10
    minx, miny, maxx, maxy = size, size, -1, -1
    for y in range(size):
        for x in range(size):
            if px[y * size + x] > thr:
                if x < minx: minx = x
                if x > maxx: maxx = x
                if y < miny: miny = y
                if y > maxy: maxy = y
    if maxx < minx or maxy < miny:
        return px
    bw, bh = maxx - minx + 1, maxy - miny + 1
    target_box = max(4.0, size * (22.0 / 28.0))
    scale = target_box / max(bw, bh)
    ox = (size - bw * scale) * 0.5
    oy = (size - bh * scale) * 0.5

    def smp(x: int, y: int) -> float:
        return float(px[y * size + x]) if 0 <= x < size and 0 <= y < size else 0.0

    out = [0] * (size * size)
    for ty in range(size):
        for tx in range(size):
            sx = (tx + 0.5 - ox) / scale + minx - 0.5
            sy = (ty + 0.5 - oy) / scale + miny - 0.5
            x0, y0 = math.floor(sx), math.floor(sy)
            wx, wy = sx - x0, sy - y0
            v = (smp(x0, y0) * (1.0 - wx) + smp(x0 + 1, y0) * wx) * (1.0 - wy) + (
                smp(x0, y0 + 1) * (1.0 - wx) + smp(x0 + 1, y0 + 1) * wx
            ) * wy
            out[ty * size + tx] = max(0, min(255, int(round(v))))
    return out


def generate_demo_dataset(out_dir: str, size: int = 28, per_class: int = 80) -> None:
    """Zero-dependency synthetic 3-class geometric dataset (circle, square, triangle)."""
    def draw_seg(buf, x0, y0, x1, y1, rad):
        steps = max(1, int(math.hypot(x1 - x0, y1 - y0) * 3))
        for s in range(steps + 1):
            t = s / steps
            cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            for y in range(size):
                for x in range(size):
                    d = math.hypot(x - cx, y - cy)
                    if d < rad:
                        v = 1.0 - (d / rad) ** 2.0
                        if v > buf[y * size + x]:
                            buf[y * size + x] = v

    for cls_name in ("circle", "square", "triangle"):
        cls_dir = os.path.join(out_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(per_class):
            buf = [0.0] * (size * size)
            # Deterministic natural variation (jitter in center, scale, stroke width)
            jx = ((i * 37) % 11 - 5) * 0.28
            jy = ((i * 53) % 11 - 5) * 0.28
            sc = 0.82 + ((i * 19) % 10) * 0.03
            rad = (size / 28.0) * (1.35 + ((i * 13) % 5) * 0.14)
            cx = size * 0.5 + jx
            cy = size * 0.5 + jy
            r = size * 0.34 * sc

            if cls_name == "circle":
                nseg = 28
                for k in range(nseg):
                    a0 = 2.0 * math.pi * k / nseg
                    a1 = 2.0 * math.pi * (k + 1) / nseg
                    draw_seg(buf, cx + r * math.cos(a0), cy + r * math.sin(a0),
                             cx + r * math.cos(a1), cy + r * math.sin(a1), rad)
            elif cls_name == "square":
                pts = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
                for k in range(4):
                    x0, y0 = pts[k]
                    x1, y1 = pts[(k + 1) % 4]
                    draw_seg(buf, x0, y0, x1, y1, rad)
            elif cls_name == "triangle":
                pts = [(cx, cy - r * 1.1), (cx + r * 1.05, cy + r * 0.9), (cx - r * 1.05, cy + r * 0.9)]
                for k in range(3):
                    x0, y0 = pts[k]
                    x1, y1 = pts[(k + 1) % 3]
                    draw_seg(buf, x0, y0, x1, y1, rad)

            px = [int(round(min(1.0, v) * 255.0)) for v in buf]
            write_pgm(os.path.join(cls_dir, f"{cls_name}_{i:03d}.pgm"), size, size, px)

    print(f"[DEMO]   Generated {3 * per_class} PGM images ({size}x{size}) across 3 classes in: {out_dir}")
    print(f"         Classes: circle ({per_class}), square ({per_class}), triangle ({per_class})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert custom class-folder images into GLLM-ready PGM datasets.")
    ap.add_argument("--input", help="Input folder containing one subfolder per class label")
    ap.add_argument("--output", help="Output folder where WxH binary PGM (P5) files will be written")
    ap.add_argument("--size", type=int, default=28, help="Output square image width & height (default: 28)")
    ap.add_argument("--invert", action="store_true", help="Invert intensities (use for black ink on white paper)")
    ap.add_argument("--auto-invert", action="store_true", help="Auto-detect bright backgrounds and invert to dark-bg")
    ap.add_argument("--center", action="store_true", help="Center & aspect-normalize ink to canonical 22/28 frame")
    ap.add_argument("--generate-demo", metavar="OUT_DIR", help="Generate a 3-class geometric demo dataset (zero deps)")
    ap.add_argument("--demo-count", type=int, default=80, help="Images per class for --generate-demo (default: 80)")
    args = ap.parse_args()

    if args.generate_demo:
        generate_demo_dataset(args.generate_demo, size=args.size, per_class=args.demo_count)
        return 0

    if not args.input or not args.output:
        ap.print_help()
        return 1

    if not os.path.isdir(args.input):
        print(f"[ERROR] Input directory not found: {args.input}", file=sys.stderr)
        return 1

    class_dirs = sorted(
        d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d)) and not d.startswith(".")
    )
    if not class_dirs:
        print(f"[ERROR] No class subdirectories found inside: {args.input}", file=sys.stderr)
        print("        Expected structure: <input>/<class_label>/*.png (or .jpg/.bmp/.pgm/.ppm)", file=sys.stderr)
        return 1

    total = 0
    for cls in class_dirs:
        src_cls_dir = os.path.join(args.input, cls)
        dst_cls_dir = os.path.join(args.output, cls)
        files = sorted(
            f for f in os.listdir(src_cls_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
        )
        cls_cnt = 0
        for idx, fname in enumerate(files):
            src_path = os.path.join(src_cls_dir, fname)
            px = load_grayscale_resized(src_path, args.size)
            px = maybe_invert(px, args.invert, args.auto_invert)
            if args.center:
                px = center_canonical(px, args.size)
            dst_path = os.path.join(dst_cls_dir, f"{idx:06d}.pgm")
            write_pgm(dst_path, args.size, args.size, px)
            cls_cnt += 1
            total += 1
        print(f"  [PREP] Class '{cls}': {cls_cnt} images -> {args.size}x{args.size} PGM")

    print(f"[DONE]   Prepared {total} images across {len(class_dirs)} classes in: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
