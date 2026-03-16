import pandas as pd
import fasttext
from pathlib import Path

# Paths
INPUT_CSV = "data/filtered_dataset.csv"
FASTTEXT_MODEL = "lid.176.bin"
OUTPUT_CSV = "data/filtered_spanish_dataset.csv"

# Load model
model = fasttext.load_model(FASTTEXT_MODEL)

def predict_language(text: str):
    """
    Returns:
        lang (str): predicted language code, e.g. 'es', 'la'
        prob (float): confidence score
    """
    if not isinstance(text, str):
        text = ""
    text = text.replace("\n", " ").strip()

    if not text:
        return None, 0.0

    labels, probs = model.predict(text, k=1)
    lang = labels[0].replace("__label__", "")
    prob = float(probs[0])
    return lang, prob

def main():
    df = pd.read_csv(INPUT_CSV)

    # Predict language from ground-truth text
    preds = df["gt_text"].apply(predict_language)
    df["pred_lang"] = preds.apply(lambda x: x[0])
    df["pred_prob"] = preds.apply(lambda x: x[1])

    # Keep only Spanish
    spanish_df = df[df["pred_lang"] == "es"].copy()

    # Optional: sort by confidence descending
    spanish_df = spanish_df.sort_values("pred_prob", ascending=False)

    # Save
    spanish_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Spanish rows: {len(spanish_df)}")
    print(f"Saved to: {Path(OUTPUT_CSV).resolve()}")

    if len(spanish_df) > 0:
        cols_to_show = ["gt_text", "ocr_text", "pred_lang", "pred_prob"]
        existing_cols = [c for c in cols_to_show if c in spanish_df.columns]
        print("\nSample Spanish rows:")
        print(spanish_df[existing_cols].head(20).to_string(index=False))

if __name__ == "__main__":
    main()