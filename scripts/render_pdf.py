#!/usr/bin/env python3
"""
Render all PDFs in a folder into per-page PNG images.

Inputs:
  1) datapath (folder containing .pdf files)
  2) output folder
  3) DPI (render resolution)

Output:
  output_folder/
    <pdf_stem_1>/
      1.png
      2.png
      ...
    <pdf_stem_2>/
      1.png
      ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def safe_stem(p: Path) -> str:
    # Avoid weird chars in folder names
    s = p.stem.strip()
    return "".join(c if (c.isalnum() or c in "._- ") else "_" for c in s) or "pdf"


def render_pdf_to_pngs(pdf_path: Path, out_dir: Path, dpi: int, max_pages: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = dpi / 72.0  # PDFs use 72 points per inch
    mat = fitz.Matrix(scale, scale)

    with fitz.open(pdf_path) as doc:
        for i in range(doc.page_count):
            if max_pages is not None and i >= max_pages:
                break
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)  # alpha=False -> no transparency
            out_path = out_dir / f"{i+1}.png"  # 1-based page numbering
            pix.save(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PDFs to per-page PNGs.")
    parser.add_argument("datapath", type=str, help="Folder containing PDF files.")
    parser.add_argument("output_folder", type=str, help="Output folder.")
    parser.add_argument("--dpi", type=int, default=400, help="Render DPI (default: 400).")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum number of pages to render per PDF.")
    args = parser.parse_args()

    datapath = Path(args.datapath).expanduser().resolve()
    output_folder = Path(args.output_folder).expanduser().resolve()
    dpi = int(args.dpi)
    max_pages = args.max_pages

    if not datapath.exists() or not datapath.is_dir():
        raise SystemExit(f"Datapath is not a directory: {datapath}")
    if dpi <= 0:
        raise SystemExit("DPI must be a positive integer.")

    output_folder.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(datapath.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in: {datapath}")

    for pdf in pdfs:
        pdf_out = output_folder / safe_stem(pdf)
        try:
            render_pdf_to_pngs(pdf, pdf_out, dpi, max_pages=max_pages)
            print(f"[OK] {pdf.name} -> {pdf_out} ({dpi} DPI)")
        except Exception as e:
            print(f"[ERR] Failed on {pdf.name}: {e}")


if __name__ == "__main__":
    main()