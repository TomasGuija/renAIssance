#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# python-docx
from docx import Document

Image.MAX_IMAGE_PIXELS = None  # or a large number

# -------------------------
# Kraken segmentation utils (same logic you had)
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


def run_kraken_segment(png_path: Path, out_json: Path, text_direction: str) -> None:
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
# DOCX parsing: "PDF p2" sections
# -------------------------
PAGE_RE = re.compile(r"^\s*PDF\s+p\s*(\d+)\s*$", re.IGNORECASE)
LINEBREAK_SPLIT_RE = re.compile(r"[\n\v]+")

def parse_docx_pages(docx_path: Path) -> Dict[int, List[str]]:
    """
    Returns {page_num: [gt_line1, gt_line2, ...]}
    Splits both by paragraphs AND by manual line breaks inside paragraphs.
    """
    doc = Document(str(docx_path))
    pages: Dict[int, List[str]] = {}

    current_page: Optional[int] = None
    for para in doc.paragraphs:
        raw = para.text or ""
        txt = raw.strip()
        if not txt:
            continue

        m = PAGE_RE.match(txt)
        if m:
            current_page = int(m.group(1))
            pages.setdefault(current_page, [])
            continue

        if current_page is None:
            # still in NOTES or header area
            continue

        # Split paragraph into physical lines if it contains manual line breaks
        parts = [p.strip() for p in LINEBREAK_SPLIT_RE.split(raw) if p.strip()]
        pages[current_page].extend(parts)

    # prune empty
    pages = {k: v for k, v in pages.items() if any(s.strip() for s in v)}
    return pages

# -------------------------
# Finding page images
# -------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def find_page_image(pdf_pages_dir: Path, page_num: int) -> Optional[Path]:
    """
    Heuristic:
      1) look for files with the page number in the stem (p2, page2, 0002, etc.)
      2) else: sort images and use (page_num - 1) index
    """
    imgs = [p for p in pdf_pages_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not imgs:
        return None
    imgs.sort(key=lambda x: str(x).lower())

    # Try pattern match by number in stem
    # accept: p2, _2, -2, 0002, page_0002, etc.
    candidates: List[Path] = []
    num = str(page_num)
    num4 = f"{page_num:04d}"
    num3 = f"{page_num:03d}"
    patterns = [
        re.compile(rf"(?:^|[^0-9]){re.escape(num4)}(?:[^0-9]|$)"),
        re.compile(rf"(?:^|[^0-9]){re.escape(num3)}(?:[^0-9]|$)"),
        re.compile(rf"(?:^|[^0-9]){re.escape(num)}(?:[^0-9]|$)"),
        re.compile(rf"(?:^|[^a-z0-9])p{re.escape(num)}(?:[^a-z0-9]|$)", re.IGNORECASE),
    ]
    for p in imgs:
        stem = p.stem
        if any(rx.search(stem) for rx in patterns):
            candidates.append(p)

    if candidates:
        candidates.sort(key=lambda x: len(x.stem))  # prefer tighter match
        return candidates[0]

    # Fallback: assume ordered list corresponds to pages starting at 1
    idx = page_num - 1
    if 0 <= idx < len(imgs):
        return imgs[idx]

    return None


# -------------------------
# OCR
# -------------------------
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\sáéíóúüñçàèìòùâêîôûäëïöü]+", "", s, flags=re.UNICODE)
    return s.strip()


def ocr_line_pytesseract(img: Image.Image, lang: str = "eng") -> str:
    try:
        import pytesseract
    except ImportError:
        raise RuntimeError("pytesseract is not installed. `pip install pytesseract` and install Tesseract OCR on Windows.")
    # PSM 7: treat as a single line
    return pytesseract.image_to_string(img, lang=lang, config="--psm 7").strip()


# -------------------------
# Alignment: monotonic DP (sequence alignment with skips)
# -------------------------
@dataclass
class Match:
    crop_i: int          # index in crops list
    gt_i: int            # index in gt list
    score: float         # similarity 0..1
    ocr_text: str
    gt_text: str


def sim(a: str, b: str) -> float:
    a2 = normalize_text(a)
    b2 = normalize_text(b)
    if not a2 and not b2:
        return 1.0
    if not a2 or not b2:
        return 0.0
    return SequenceMatcher(None, a2, b2).ratio()


def align_sequences(
    ocr_texts: List[str],
    gt_lines: List[str],
    skip_cost: float = 0.65,
) -> Tuple[List[Match], List[int], List[int]]:
    """
    Align ocr_texts (length M) to gt_lines (length N), monotonic, allowing skips on either side.
    Returns:
      - matches: list of (crop_i, gt_i, score)
      - unassigned_crops: indices in ocr_texts not matched
      - unassigned_gt: indices in gt_lines not matched
    """
    M, N = len(ocr_texts), len(gt_lines)
    # cost = 1 - similarity
    cost = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            cost[i, j] = 1.0 - sim(ocr_texts[i], gt_lines[j])

    # DP
    dp = np.full((M + 1, N + 1), np.inf, dtype=np.float32)
    back = np.zeros((M + 1, N + 1), dtype=np.int8)  # 0 diag(match), 1 up(skip crop), 2 left(skip gt)
    dp[0, 0] = 0.0

    for i in range(1, M + 1):
        dp[i, 0] = dp[i - 1, 0] + skip_cost
        back[i, 0] = 1
    for j in range(1, N + 1):
        dp[0, j] = dp[0, j - 1] + skip_cost
        back[0, j] = 2

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            d_match = dp[i - 1, j - 1] + cost[i - 1, j - 1]
            d_skip_crop = dp[i - 1, j] + skip_cost
            d_skip_gt = dp[i, j - 1] + skip_cost
            best = d_match
            b = 0
            if d_skip_crop < best:
                best = d_skip_crop
                b = 1
            if d_skip_gt < best:
                best = d_skip_gt
                b = 2
            dp[i, j] = best
            back[i, j] = b

    # backtrack
    i, j = M, N
    matches: List[Match] = []
    matched_crops = set()
    matched_gt = set()
    while i > 0 or j > 0:
        b = int(back[i, j])
        if b == 0 and i > 0 and j > 0:
            si = 1.0 - float(cost[i - 1, j - 1])
            matches.append(Match(
                crop_i=i - 1,
                gt_i=j - 1,
                score=si,
                ocr_text=ocr_texts[i - 1],
                gt_text=gt_lines[j - 1],
            ))
            matched_crops.add(i - 1)
            matched_gt.add(j - 1)
            i -= 1
            j -= 1
        elif b == 1 and i > 0:
            i -= 1
        else:
            j -= 1

    matches.reverse()
    unassigned_crops = [k for k in range(M) if k not in matched_crops]
    unassigned_gt = [k for k in range(N) if k not in matched_gt]
    return matches, unassigned_crops, unassigned_gt


# -------------------------
# Main pipeline
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages_root", required=True, type=str, help="Root folder with subfolders per PDF, containing page images.")
    ap.add_argument("--docx_root", required=True, type=str, help="Root folder containing .docx (one per PDF).")
    ap.add_argument("--out_root", required=True, type=str, help="Output dataset root.")
    ap.add_argument("--text_direction", default="horizontal-lr", type=str)
    ap.add_argument("--pad", default=2, type=int)

    ap.add_argument("--ocr_engine", default="tesseract", choices=["tesseract"], help="Pretrained OCR used for draft alignment.")
    ap.add_argument("--tess_lang", default="eng", type=str, help="Tesseract language code(s), e.g. 'eng' or 'spa' or 'eng+spa'.")
    ap.add_argument("--min_score", default=0.15, type=float, help="Minimum similarity score to accept match into training CSV.")
    ap.add_argument("--skip_cost", default=0.85, type=float, help="Alignment skip cost (higher => fewer skips).")

    args = ap.parse_args()

    pages_root = Path(args.pages_root).expanduser().resolve()
    docx_root = Path(args.docx_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    images_out = out_root / "images"
    interm_out = out_root / "intermediate"
    images_out.mkdir(parents=True, exist_ok=True)
    interm_out.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "dataset.csv"
    unassigned_path = out_root / "unassigned.jsonl"
    h5_path = out_root / "dataset.h5"

    # Collect docx files
    docx_files = sorted(docx_root.glob("*.docx"), key=lambda p: p.name.lower())
    if not docx_files:
        print(f"No .docx found in {docx_root}", file=sys.stderr)
        sys.exit(1)

    # CSV header
    csv_fields = [
        "pdf_id", "page_num", "page_id",
        "crop_rel", "gt_text", "ocr_text", "score",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "kraken_line_i",
        "page_image_rel",
    ]

    rows_for_h5: List[Dict[str, Any]] = []
    unassigned_records: List[Dict[str, Any]] = []

    csv_exists = csv_path.exists()
    write_header = (not csv_exists) or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()

        for docx_path in docx_files:
            pdf_id = docx_path.stem
            pdf_pages_dir = pages_root / pdf_id
            if not pdf_pages_dir.exists():
                print(f"[WARN] No pages folder for {pdf_id}: expected {pdf_pages_dir}")
                continue

            page_to_gt = parse_docx_pages(docx_path)
            if not page_to_gt:
                print(f"[WARN] No 'PDF pN' sections found in {docx_path.name}")
                continue


            for page_num, gt_lines in sorted(page_to_gt.items()):
                page_img = find_page_image(pdf_pages_dir, page_num)
                if page_img is None:
                    unassigned_records.append({
                        "type": "missing_page_image",
                        "pdf_id": pdf_id,
                        "page_num": page_num,
                        "docx": str(docx_path),
                        "gt_lines": gt_lines,
                    })
                    print(f"[WARN] Missing page image for {pdf_id} page {page_num}")
                    continue

                page_id = page_img.stem
                page_image_rel = str(page_img.relative_to(pages_root)).replace("\\", "/")

                # Segment + crops to intermediate folder
                page_intermediate = interm_out / pdf_id / f"p{page_num:04d}_{page_id}"
                out_page_dir = images_out / pdf_id / f"p{page_num:04d}_{page_id}"

                # GUARD: skip if crops already exist for this page
                if out_page_dir.exists():
                    crop_files = list(out_page_dir.glob("line_*.png"))
                    if crop_files:
                        print(f"[SKIP] {pdf_id} p{page_num}: crops already exist ({len(crop_files)} lines)")
                        continue

                page_intermediate.mkdir(parents=True, exist_ok=True)
                seg_json = page_intermediate / "segmentation.json"
                if not seg_json.exists():
                    run_kraken_segment(page_img, seg_json, text_direction=args.text_direction)

                seg = json.loads(seg_json.read_text(encoding="utf-8"))
                lines = extract_lines_from_kraken_json(seg)

                if not lines:
                    unassigned_records.append({
                        "type": "no_lines_segmented",
                        "pdf_id": pdf_id,
                        "page_num": page_num,
                        "page_image_rel": page_image_rel,
                        "gt_lines": gt_lines,
                    })
                    print(f"[WARN] No lines segmented for {pdf_id} page {page_num}")
                    continue

                # Load page image once
                page_rgb = Image.open(page_img).convert("RGB")
                page_np = np.array(page_rgb)
                h, w = page_np.shape[:2]

                # Crop all lines, OCR them
                crop_paths: List[Path] = []
                crop_bboxes: List[Tuple[int, int, int, int]] = []
                kraken_is: List[int] = []
                ocr_texts: List[str] = []

                out_page_dir.mkdir(parents=True, exist_ok=True)

                for k, ln in enumerate(lines, start=1):
                    x1, y1, x2, y2 = ln["bbox"]
                    x1 -= args.pad
                    y1 -= args.pad
                    x2 += args.pad
                    y2 += args.pad
                    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)

                    crop_arr = page_np[y1:y2, x1:x2]
                    crop_img = Image.fromarray(crop_arr)

                    crop_path = out_page_dir / f"line_{k:04d}.png"
                    crop_img.save(crop_path)

                    # OCR draft
                    if args.ocr_engine == "tesseract":
                        ocr = ocr_line_pytesseract(crop_img, lang=args.tess_lang)
                    else:
                        ocr = ""

                    crop_paths.append(crop_path)
                    crop_bboxes.append((x1, y1, x2, y2))
                    kraken_is.append(int(ln["i"]))
                    ocr_texts.append(ocr)

                # Align OCR lines to GT lines
                matches, unassigned_crops, unassigned_gt = align_sequences(
                    ocr_texts=ocr_texts,
                    gt_lines=gt_lines,
                    skip_cost=args.skip_cost,
                )

                debug_path = out_page_dir / "ocr_debug.txt"

                # Build quick lookup for matched lines
                match_by_crop = {m.crop_i: m for m in matches}

                with debug_path.open("w", encoding="utf-8") as fdbg:
                    for i, ocr_text in enumerate(ocr_texts):
                        line_id = f"{i+1:04d}"
                        bbox = crop_bboxes[i]
                        bbox_str = f"bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"

                        if i in match_by_crop:
                            m = match_by_crop[i]
                            status = "MATCH" if m.score >= args.min_score else "LOW_SCORE"
                            fdbg.write(
                                f"{line_id}\t{status}\t{bbox_str}\n"
                                f"OCR: {ocr_text}\n"
                                f"GT : {m.gt_text}\n"
                                f"Score: {m.score:.4f}\n\n"
                            )
                        else:
                            fdbg.write(
                                f"{line_id}\tUNASSIGNED\t{bbox_str}\n"
                                f"OCR: {ocr_text}\n\n"
                            )

                # Save accepted matches to CSV + H5 rows
                for m in matches:
                    if m.score < args.min_score:
                        continue
                    crop_path = crop_paths[m.crop_i]
                    bbox = crop_bboxes[m.crop_i]
                    crop_rel = str(crop_path.relative_to(out_root)).replace("\\", "/")

                    row = {
                        "pdf_id": pdf_id,
                        "page_num": page_num,
                        "page_id": page_id,
                        "crop_rel": crop_rel,
                        "gt_text": m.gt_text,
                        "ocr_text": m.ocr_text,
                        "score": f"{m.score:.4f}",
                        "bbox_x1": bbox[0],
                        "bbox_y1": bbox[1],
                        "bbox_x2": bbox[2],
                        "bbox_y2": bbox[3],
                        "kraken_line_i": kraken_is[m.crop_i],
                        "page_image_rel": page_image_rel,
                    }
                    writer.writerow(row)
                    rows_for_h5.append(row)

                # Record unassigned items explicitly
                if unassigned_gt:
                    unassigned_records.append({
                        "type": "unassigned_gt_lines",
                        "pdf_id": pdf_id,
                        "page_num": page_num,
                        "page_image_rel": page_image_rel,
                        "docx": str(docx_path),
                        "gt_unassigned": [{"gt_i": i, "text": gt_lines[i]} for i in unassigned_gt],
                    })

                if unassigned_crops:
                    unassigned_records.append({
                        "type": "unassigned_crops",
                        "pdf_id": pdf_id,
                        "page_num": page_num,
                        "page_image_rel": page_image_rel,
                        "crop_unassigned": [
                            {
                                "crop_i": i,
                                "crop_rel": str(crop_paths[i].relative_to(out_root)).replace("\\", "/"),
                                "ocr_text": ocr_texts[i],
                                "bbox": list(crop_bboxes[i]),
                            }
                            for i in unassigned_crops
                        ],
                    })

                print(f"{pdf_id} p{page_num}: crops={len(crop_paths)} gt={len(gt_lines)} "
                      f"matches={sum(1 for mm in matches if mm.score >= args.min_score)} "
                      f"unGT={len(unassigned_gt)} unCrops={len(unassigned_crops)}")

    # Write unassigned JSONL
    with unassigned_path.open("w", encoding="utf-8") as f:
        for rec in unassigned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Rebuild H5 from full CSV so reruns/appends remain consistent
    all_rows_for_h5: List[Dict[str, Any]] = []
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as fcsv_in:
            reader = csv.DictReader(fcsv_in)
            all_rows_for_h5 = list(reader)

    # Write HDF5 (optional but requested)
    try:
        import h5py  # type: ignore
        with h5py.File(h5_path, "w") as h5:
            # store as variable-length UTF-8 strings
            dt = h5py.string_dtype(encoding="utf-8")
            h5.create_dataset("pdf_id", data=np.array([r["pdf_id"] for r in all_rows_for_h5], dtype=object), dtype=dt)
            h5.create_dataset("page_num", data=np.array([int(r["page_num"]) for r in all_rows_for_h5], dtype=np.int32))
            h5.create_dataset("page_id", data=np.array([r["page_id"] for r in all_rows_for_h5], dtype=object), dtype=dt)
            h5.create_dataset("crop_rel", data=np.array([r["crop_rel"] for r in all_rows_for_h5], dtype=object), dtype=dt)
            h5.create_dataset("gt_text", data=np.array([r["gt_text"] for r in all_rows_for_h5], dtype=object), dtype=dt)
            h5.create_dataset("ocr_text", data=np.array([r["ocr_text"] for r in all_rows_for_h5], dtype=object), dtype=dt)
            h5.create_dataset("score", data=np.array([float(r["score"]) for r in all_rows_for_h5], dtype=np.float32))
            h5.create_dataset("bbox", data=np.array([[r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]] for r in all_rows_for_h5], dtype=np.int32))
            h5.create_dataset("kraken_line_i", data=np.array([int(r["kraken_line_i"]) for r in all_rows_for_h5], dtype=np.int32))
            h5.create_dataset("page_image_rel", data=np.array([r["page_image_rel"] for r in all_rows_for_h5], dtype=object), dtype=dt)
    except ImportError:
        print("[WARN] h5py not installed; skipping dataset.h5. Install with: pip install h5py")

    print(f"\nWrote:\n  {csv_path}\n  {unassigned_path}\n  {h5_path} (if h5py installed)\n  images under: {images_out}")


if __name__ == "__main__":
    main()