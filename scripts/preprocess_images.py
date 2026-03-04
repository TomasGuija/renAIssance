#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def preprocess_image(
    image: Image.Image,
    target_height: int,
    p_low: float,
    p_high: float,
    gamma: float,
    max_width: int,
) -> Image.Image:
    image = image.convert("L")
    arr = np.asarray(image, dtype=np.float32)

    low = float(np.percentile(arr, p_low))
    high = float(np.percentile(arr, p_high))

    if high <= low + 1e-6:
        stretched = arr
    else:
        stretched = (arr - low) / (high - low)
        stretched = np.clip(stretched, 0.0, 1.0) * 255.0

    if abs(gamma - 1.0) > 1e-6:
        norm = np.clip(stretched / 255.0, 0.0, 1.0)
        stretched = np.power(norm, gamma) * 255.0

    stretched = np.clip(stretched, 0.0, 255.0).astype(np.uint8)
    out = Image.fromarray(stretched, mode="L")

    width, height = out.size
    if height <= 0:
        return out

    new_width = max(1, int(round(width * (target_height / float(height)))))
    new_width = min(new_width, max_width)

    if new_width != width or target_height != height:
        out = out.resize((new_width, target_height), Image.Resampling.BICUBIC)

    return out


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess images offline and mirror input folder structure to output folder."
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        default="data/images",
        type=str,
        help="Root folder with original PNG line images (default: data/images).",
    )
    parser.add_argument("output_root", type=str, help="Root folder where preprocessed data is written.")
    parser.add_argument("--height", type=int, default=32, help="Target image height (default: 32).")
    parser.add_argument("--max-width", type=int, default=1600, help="Maximum output width after resize (default: 1600).")
    parser.add_argument("--p-low", type=float, default=2.0, help="Lower percentile for contrast clamping (default: 2).")
    parser.add_argument("--p-high", type=float, default=98.0, help="Upper percentile for contrast clamping (default: 98).")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction factor (default: 1.0).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files already present in output.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"Input root is not a directory: {input_root}")
    if args.height <= 0:
        raise SystemExit("--height must be a positive integer")
    if args.max_width <= 0:
        raise SystemExit("--max-width must be a positive integer")
    if not (0.0 <= args.p_low < args.p_high <= 100.0):
        raise SystemExit("Percentiles must satisfy 0 <= p-low < p-high <= 100")
    if args.gamma <= 0:
        raise SystemExit("--gamma must be > 0")

    output_root.mkdir(parents=True, exist_ok=True)

    processed_images = 0
    copied_files = 0
    skipped_existing = 0
    failed = 0

    for source_path in input_root.rglob("*"):
        if source_path.is_dir():
            continue

        if is_relative_to(source_path, output_root):
            continue

        if source_path.suffix.lower() != ".png":
            continue

        relative = source_path.relative_to(input_root)
        dest_path = output_root / relative
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        try:
            with Image.open(source_path) as img:
                out = preprocess_image(
                    image=img,
                    target_height=args.height,
                    p_low=args.p_low,
                    p_high=args.p_high,
                    gamma=args.gamma,
                    max_width=args.max_width,
                )
                out.save(dest_path)
            processed_images += 1
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {source_path} -> {dest_path}: {exc}")

    print("Preprocessing complete")
    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Images processed: {processed_images}")
    print(f"Non-image files copied: {copied_files}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
