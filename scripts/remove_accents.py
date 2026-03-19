import csv
import unicodedata
from pathlib import Path

INPUT_CSV = "data/filtered_spanish_dataset.csv"
OUTPUT_CSV = "data/filtered_spanish_dataset_no_accents.csv"
TEXT_COLUMN = "gt_text"


def remove_vowel_accents(text: str) -> str:
    """
    Remove diacritics from vowels only, preserving ñ / Ñ and other consonants.
    """
    if not isinstance(text, str):
        return text

    result = []

    for ch in text:
        if ch in ("ñ", "Ñ"):
            result.append(ch)
            continue

        # Decompose unicode character into base + combining marks
        decomposed = unicodedata.normalize("NFD", ch)
        base = decomposed[0]

        # Only strip accents if the base character is a vowel
        if base in "aeiouAEIOU":
            stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
            result.append(stripped)
        else:
            result.append(ch)

    return "".join(result)


def process_csv(input_csv: str, output_csv: str, text_column: str = "gt_text"):
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    with input_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError("Input CSV has no header.")

        if text_column not in fieldnames:
            raise ValueError(f"Column '{text_column}' not found in CSV headers: {fieldnames}")

        rows = []
        for row in reader:
            row[text_column] = remove_vowel_accents(row.get(text_column, ""))
            rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Saved cleaned CSV to: {output_path}")


if __name__ == "__main__":
    process_csv(INPUT_CSV, OUTPUT_CSV, TEXT_COLUMN)