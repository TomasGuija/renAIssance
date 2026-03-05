#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def filter_rows(
    dataset_csv: Path,
    output_csv: Path,
    removed_csv: Path,
    gt_column: str,
    contains_list: list[str],
) -> tuple[int, int]:
    with dataset_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if gt_column not in fieldnames:
            raise ValueError(f"Column '{gt_column}' not found in CSV. Available: {fieldnames}")
        rows = list(reader)

    kept_rows = []
    removed_rows = []

    for row in rows:
        text = row.get(gt_column) or ""
        if any(token in text for token in contains_list):
            removed_rows.append(row)
        else:
            kept_rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    removed_csv.parent.mkdir(parents=True, exist_ok=True)
    with removed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(removed_rows)

    return len(kept_rows), len(removed_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter dataset rows by GT substring and write kept/removed CSV files."
    )
    parser.add_argument(
        "--dataset_csv",
        type=str,
        default="data/dataset.csv",
        help="Input dataset CSV path",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV for kept rows (default: <dataset>.filtered.csv)",
    )
    parser.add_argument(
        "--removed_csv",
        type=str,
        default=None,
        help="Output CSV for removed rows (default: <dataset>.filtered_removed.csv)",
    )
    parser.add_argument(
        "--gt_column",
        type=str,
        default="gt_text",
        help="GT text column name (default: gt_text)",
    )
    parser.add_argument(
        "--contains",
        type=str,
        nargs="+",
        default=["...", "…"],
        help="One or more substrings. Rows are removed if GT contains any of them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_csv = Path(args.dataset_csv).expanduser().resolve()
    if not dataset_csv.exists():
        raise SystemExit(f"Input CSV not found: {dataset_csv}")

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
    else:
        output_csv = dataset_csv.with_name(dataset_csv.stem + ".filtered.csv")

    if args.removed_csv:
        removed_csv = Path(args.removed_csv).expanduser().resolve()
    else:
        removed_csv = dataset_csv.with_name(dataset_csv.stem + ".filtered_removed.csv")

    kept, removed = filter_rows(
        dataset_csv=dataset_csv,
        output_csv=output_csv,
        removed_csv=removed_csv,
        gt_column=args.gt_column,
        contains_list=args.contains,
    )

    print(f"Input:   {dataset_csv}")
    print(f"Output:  {output_csv}")
    print(f"Removed: {removed_csv}")
    print(f"Kept rows: {kept}")
    print(f"Removed rows: {removed}")


if __name__ == "__main__":
    main()
