#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image


# -------------------------
# Geometry helpers
# -------------------------
def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(1, min(int(x2), w))
    y2 = max(1, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def bbox_from_boundary(boundary: Any) -> Tuple[int, int, int, int]:
    pts = np.asarray(boundary, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def extract_lines_from_kraken_json(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    lines = seg.get("lines", [])
    out: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines):
        bbox = None
        if isinstance(ln, dict):
            if "bbox" in ln and ln["bbox"] is not None:
                bb = ln["bbox"]
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    bbox = tuple(map(int, bb))
            elif "boundary" in ln and ln["boundary"] is not None:
                bbox = bbox_from_boundary(ln["boundary"])
        if bbox is None:
            continue
        out.append({"i": i, "bbox": bbox, "raw": ln})
    return out


# -------------------------
# Kraken invocation
# -------------------------
def run_kraken_segment(png_path: Path, out_json: Path, text_direction: str = "horizontal-lr") -> None:
    cmd = [
        "kraken",
        "-i", str(png_path),
        str(out_json),
        "segment",
        "-bl",
        "-d", text_direction,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("ERROR: 'kraken' command not found. Did you `pip install kraken` in this venv?", file=sys.stderr)
        raise
    except subprocess.CalledProcessError as e:
        print(f"ERROR: kraken failed on {png_path}", file=sys.stderr)
        print("STDOUT:\n", e.stdout, file=sys.stderr)
        print("STDERR:\n", e.stderr, file=sys.stderr)
        raise


# -------------------------
# Metadata (for next step)
# -------------------------
def sha1_of_path(p: Path) -> str:
    return hashlib.sha1(str(p).encode("utf-8")).hexdigest()


@dataclass
class LineMeta:
    pdf_id: str
    page_id: str
    page_rel: str
    page_w: int
    page_h: int
    line_index: int  # 1-based in our naming
    bbox: Tuple[int, int, int, int]
    crop_rel: str
    seg_rel: str
    kraken_line_i: int  # original index in kraken JSON lines list

    def to_json(self) -> str:
        d = {
            "pdf_id": self.pdf_id,
            "page_id": self.page_id,
            "page_rel": self.page_rel,
            "page_w": self.page_w,
            "page_h": self.page_h,
            "line_index": self.line_index,
            "bbox": list(self.bbox),
            "crop_rel": self.crop_rel,
            "seg_rel": self.seg_rel,
            "kraken_line_i": self.kraken_line_i,
        }
        return json.dumps(d, ensure_ascii=False)


# -------------------------
# Main processing
# -------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def iter_page_images(pdf_folder: Path) -> List[Path]:
    imgs: List[Path] = []
    for p in pdf_folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    # stable order: by path (usually matches page_0001.png naming)
    imgs.sort(key=lambda x: str(x).lower())
    return imgs


def process_page(
    png_path: Path,
    out_page_dir: Path,
    text_direction: str,
    pad: int,
    meta_out,
    pdf_id: str,
    page_id: str,
    page_rel: str,
    out_root: Path,
) -> int:
    out_page_dir.mkdir(parents=True, exist_ok=True)

    page = Image.open(png_path).convert("RGB")
    page_np = np.array(page)
    h, w = page_np.shape[:2]

    seg_json = out_page_dir / "segmentation.json"
    if not seg_json.exists():
        run_kraken_segment(png_path, seg_json, text_direction=text_direction)

    seg = json.loads(seg_json.read_text(encoding="utf-8"))
    lines = extract_lines_from_kraken_json(seg)
    if not lines:
        return 0

    n_saved = 0
    for k, ln in enumerate(lines, start=1):
        x1, y1, x2, y2 = ln["bbox"]
        x1 -= pad
        y1 -= pad
        x2 += pad
        y2 += pad
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)

        crop = page_np[y1:y2, x1:x2]
        out_path = out_page_dir / f"line_{k:04d}.png"
        Image.fromarray(crop).save(out_path)

        crop_rel = str(out_path.relative_to(out_root)).replace("\\", "/")
        seg_rel = str(seg_json.relative_to(out_root)).replace("\\", "/")

        meta = LineMeta(
            pdf_id=pdf_id,
            page_id=page_id,
            page_rel=page_rel,
            page_w=w,
            page_h=h,
            line_index=k,
            bbox=(x1, y1, x2, y2),
            crop_rel=crop_rel,
            seg_rel=seg_rel,
            kraken_line_i=int(ln["i"]),
        )
        meta_out.write(meta.to_json() + "\n")
        n_saved += 1

    return n_saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, type=str, help="Root folder with subfolders per PDF.")
    ap.add_argument("--out_root", required=True, type=str, help="Root folder for cropped lines.")
    ap.add_argument("--text_direction", default="horizontal-lr", type=str)
    ap.add_argument("--pad", default=2, type=int)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing line crops (re-crop).")
    ap.add_argument("--dry_run", action="store_true", help="List what would be processed but do nothing.")
    args = ap.parse_args()

    in_root = Path(args.in_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Global metadata file (one JSON per line crop)
    meta_path = out_root / "meta.jsonl"

    # If overwrite: remove old meta and let pages regenerate crops.
    if args.overwrite and meta_path.exists() and not args.dry_run:
        meta_path.unlink()

    pdf_folders = [p for p in in_root.iterdir() if p.is_dir()]
    pdf_folders.sort(key=lambda x: x.name.lower())

    if args.dry_run:
        print(f"[DRY RUN] Would process {len(pdf_folders)} PDF folders under: {in_root}")
        for pf in pdf_folders:
            imgs = iter_page_images(pf)
            print(f"  - {pf.name}: {len(imgs)} page images")
        return

    with meta_path.open("a", encoding="utf-8") as meta_out:
        total_pages = 0
        total_lines = 0

        for pdf_folder in pdf_folders:
            pdf_id = pdf_folder.name
            page_imgs = iter_page_images(pdf_folder)
            if not page_imgs:
                continue

            for img_path in page_imgs:
                # page_id based on filename stem; stable + human-readable
                page_id = img_path.stem

                # keep relative page path (so we can find the original image later)
                page_rel = str(img_path.relative_to(in_root)).replace("\\", "/")

                # Output: out_root/pdf_id/page_id/*
                out_page_dir = out_root / pdf_id / page_id

                if args.overwrite and out_page_dir.exists():
                    # Only delete line_*.png, keep seg json if you want faster reruns
                    for p in out_page_dir.glob("line_*.png"):
                        p.unlink(missing_ok=True)

                n = process_page(
                    png_path=img_path,
                    out_page_dir=out_page_dir,
                    text_direction=args.text_direction,
                    pad=args.pad,
                    meta_out=meta_out,
                    pdf_id=pdf_id,
                    page_id=page_id,
                    page_rel=page_rel,
                    out_root=out_root,
                )
                total_pages += 1
                total_lines += n
                print(f"{pdf_id}/{page_id}: {n} lines")

        print(f"Done. Pages: {total_pages}, line crops: {total_lines}")
        print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()