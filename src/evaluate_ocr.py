import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from Levenshtein import distance

from dataset import CsvLineDataset, AlignCollate
from model import Model
from utils import CTCLabelConverter


def read_dataset_rows(dataset_csv):
    rows = []
    with open(dataset_csv, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("crop_rel") and row.get("gt_text") is not None and row.get("pdf_id"):
                rows.append(row)
    return rows


def filter_rows_by_pdf_ids(rows, pdf_ids):
    pdf_id_set = set(pdf_ids)
    return [row for row in rows if row["pdf_id"] in pdf_id_set]


def load_model_from_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    hparams = ckpt["hyper_parameters"]
    opt = SimpleNamespace(**hparams)
    opt.input_channel = 1  # Set to 1 for grayscale input

    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model = Model(opt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, opt, converter


def compute_metrics(predictions, references):
    total_edit_distance = 0
    total_ref_chars = 0

    for pred, ref in zip(predictions, references):
        total_edit_distance += distance(pred, ref)
        total_ref_chars += len(ref)

    cer = total_edit_distance / max(total_ref_chars, 1)

    return {
        "num_samples": len(references),
        "cer": cer,
    }


def evaluate(model, converter, dataloader, device, save_predictions=None):
    predictions = []
    references = []
    prediction_rows = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            batch_size = images.size(0)

            preds = model(images)
            _, preds_index = preds.max(2)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)

            preds_str = converter.decode(preds_index, preds_size)

            for pred, ref in zip(preds_str, labels):
                predictions.append(pred)
                references.append(ref)
                prediction_rows.append({
                    "ground_truth": ref,
                    "prediction": pred,
                })

    metrics = compute_metrics(predictions, references)

    if save_predictions is not None:
        save_predictions = Path(save_predictions)
        save_predictions.parent.mkdir(parents=True, exist_ok=True)

        with open(save_predictions, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ground_truth", "prediction"])
            writer.writeheader()
            writer.writerows(prediction_rows)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR checkpoint on selected pdf_id values.")
    parser.add_argument("--dataset_csv", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory for crop_rel paths")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--pdf_ids",
        type=str,
        nargs="+",
        required=True,
        help="List of pdf_id values to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--save_predictions",
        type=str,
        default=None,
        help="Optional path to save predictions as CSV",
    )

    args = parser.parse_args()

    dataset_csv = Path(args.dataset_csv).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from: {checkpoint}")
    model, opt, converter = load_model_from_checkpoint(checkpoint, device)

    print(f"Reading dataset from: {dataset_csv}")
    rows = read_dataset_rows(dataset_csv)
    rows = filter_rows_by_pdf_ids(rows, args.pdf_ids)

    if not rows:
        raise ValueError("No rows found for the selected pdf_id values.")

    print(f"Selected pdf_ids: {args.pdf_ids}")
    print(f"Rows before dataset filtering: {len(rows)}")

    dataset = CsvLineDataset(rows, opt=opt, image_root=image_root)

    if len(dataset) == 0:
        raise ValueError("No valid samples remain after dataset filtering.")

    print(f"Usable samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=AlignCollate(imgH=opt.imgH),
    )

    metrics = evaluate(
        model=model,
        converter=converter,
        dataloader=dataloader,
        device=device,
        save_predictions=args.save_predictions,
    )

    print("\nEvaluation results")
    print("------------------")
    print(f"Samples:       {metrics['num_samples']}")
    print(f"CER:           {metrics['cer']:.4f}")

    if args.save_predictions:
        print(f"\nPredictions saved to: {args.save_predictions}")


if __name__ == "__main__":
    main()