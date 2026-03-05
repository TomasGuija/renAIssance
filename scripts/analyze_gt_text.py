#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def analyze_gt(csv_path: Path, gt_column: str = 'gt_text'):
    max_len = -1
    max_len_row = -1
    charset = set()
    total_rows = 0
    valid_rows = 0
    lengths = []

    with csv_path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if gt_column not in (reader.fieldnames or []):
            raise ValueError(f"Column '{gt_column}' not found. Available columns: {reader.fieldnames}")

        for row_idx, row in enumerate(reader, start=2):
            total_rows += 1
            text = row.get(gt_column)
            if text is None:
                continue

            valid_rows += 1
            text_len = len(text)
            lengths.append(text_len)
            if text_len > max_len:
                max_len = text_len
                max_len_row = row_idx

            charset.update(text)

    sorted_chars = sorted(charset, key=ord)
    percentiles = {}
    if lengths:
        pct_values = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        pct_results = np.percentile(np.array(lengths, dtype=np.float32), pct_values)
        percentiles = {f'p{p}': float(v) for p, v in zip(pct_values, pct_results)}

    return {
        'total_rows': total_rows,
        'valid_rows': valid_rows,
        'max_length': max_len,
        'max_length_row': max_len_row,
        'mean_length': float(np.mean(lengths)) if lengths else 0.0,
        'std_length': float(np.std(lengths)) if lengths else 0.0,
        'percentiles': percentiles,
        'charset_size': len(sorted_chars),
        'charset_chars': sorted_chars,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze GT text statistics from a dataset CSV.')
    parser.add_argument('csv_path', nargs='?', default='data/dataset.csv', help='Path to CSV file (default: data/dataset.csv)')
    parser.add_argument('--gt-column', default='gt_text', help="GT text column name (default: 'gt_text')")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f'CSV file not found: {csv_path}')

    stats = analyze_gt(csv_path=csv_path, gt_column=args.gt_column)

    print(f"CSV: {csv_path}")
    print(f"Rows (total): {stats['total_rows']}")
    print(f"Rows (with {args.gt_column}): {stats['valid_rows']}")
    print(f"Max sequence length: {stats['max_length']}")
    print(f"Row containing max length: {stats['max_length_row']}")
    print(f"Mean length: {stats['mean_length']:.2f}")
    print(f"Std length: {stats['std_length']:.2f}")
    if stats['percentiles']:
        print('Length percentiles:')
        for name, value in stats['percentiles'].items():
            print(f"  {name}: {value:.2f}")
    print(f"Unique characters: {stats['charset_size']}")
    print('Characters (ASCII-ordered):')
    print(''.join(stats['charset_chars']))


if __name__ == '__main__':
    main()
