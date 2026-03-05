from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from PIL import ImageGrab
except Exception:
    ImageGrab = None


@dataclass
class Action:
    row_index: int
    prev_decision: Optional[bool]
    prev_pos: int


class CropDialog:
    def __init__(self, parent: tk.Tk, image: Image.Image, max_width: int = 1400, max_height: int = 900) -> None:
        self.orig_w, self.orig_h = image.size
        self.scale = min(max_width / self.orig_w, max_height / self.orig_h, 1.0)
        view_w = max(1, int(self.orig_w * self.scale))
        view_h = max(1, int(self.orig_h * self.scale))

        self.preview = image.resize((view_w, view_h), Image.Resampling.LANCZOS) if self.scale < 1.0 else image.copy()
        self.tk_preview = ImageTk.PhotoImage(self.preview)

        self.result_bbox: Optional[Tuple[int, int, int, int]] = None
        self.start_x: Optional[int] = None
        self.start_y: Optional[int] = None
        self.end_x: Optional[int] = None
        self.end_y: Optional[int] = None
        self.rect_id: Optional[int] = None

        self.win = tk.Toplevel(parent)
        self.win.title("Crop Image")
        self.win.transient(parent)
        self.win.grab_set()

        tk.Label(self.win, text="Drag to select crop area, then click Apply crop").pack(padx=8, pady=(8, 4))
        self.canvas = tk.Canvas(self.win, width=view_w, height=view_h, bg="#222")
        self.canvas.pack(padx=8, pady=8)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_preview)

        controls = tk.Frame(self.win)
        controls.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Button(controls, text="Apply crop", command=self._apply).pack(side=tk.LEFT)
        tk.Button(controls, text="Cancel", command=self._cancel).pack(side=tk.LEFT, padx=6)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        self.win.wait_window()

    def _clamp(self, x: int, y: int) -> Tuple[int, int]:
        x = max(0, min(x, self.preview.width - 1))
        y = max(0, min(y, self.preview.height - 1))
        return x, y

    def _on_press(self, event: tk.Event) -> None:
        self.start_x, self.start_y = self._clamp(int(event.x), int(event.y))
        self.end_x, self.end_y = self.start_x, self.start_y
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
            outline="#ff2d2d",
            width=2,
        )

    def _on_drag(self, event: tk.Event) -> None:
        if self.start_x is None or self.start_y is None or self.rect_id is None:
            return
        self.end_x, self.end_y = self._clamp(int(event.x), int(event.y))
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, self.end_x, self.end_y)

    def _on_release(self, event: tk.Event) -> None:
        if self.start_x is None or self.start_y is None:
            return
        self.end_x, self.end_y = self._clamp(int(event.x), int(event.y))

    def _apply(self) -> None:
        if None in (self.start_x, self.start_y, self.end_x, self.end_y):
            messagebox.showwarning("No selection", "Draw a rectangle first.")
            return

        x1, x2 = sorted([int(self.start_x), int(self.end_x)])
        y1, y2 = sorted([int(self.start_y), int(self.end_y)])

        if x2 <= x1 or y2 <= y1:
            messagebox.showwarning("Invalid selection", "Selection must have non-zero width/height.")
            return

        ox1 = int(round(x1 / self.scale))
        oy1 = int(round(y1 / self.scale))
        ox2 = int(round(x2 / self.scale))
        oy2 = int(round(y2 / self.scale))

        ox1 = max(0, min(ox1, self.orig_w - 1))
        oy1 = max(0, min(oy1, self.orig_h - 1))
        ox2 = max(ox1 + 1, min(ox2, self.orig_w))
        oy2 = max(oy1 + 1, min(oy2, self.orig_h))

        self.result_bbox = (ox1, oy1, ox2, oy2)
        self.win.destroy()

    def _cancel(self) -> None:
        self.result_bbox = None
        self.win.destroy()


class DatasetReviewer:
    def __init__(
        self,
        dataset_csv: Path,
        image_root: Path,
        output_csv: Path,
        removed_csv: Path,
        session_json: Path,
        inplace: bool,
        max_width: int,
        max_height: int,
        order: str,
    ) -> None:
        self.dataset_csv = dataset_csv
        self.image_root = image_root
        self.output_csv = output_csv
        self.removed_csv = removed_csv
        self.session_json = session_json
        self.inplace = inplace
        self.max_width = max_width
        self.max_height = max_height
        self.order = order

        self.fieldnames: List[str] = []
        self.rows: List[Dict[str, str]] = []
        self.decisions: Dict[int, bool] = {}
        self.history: List[Action] = []
        self.pos = 0
        self.tk_img = None
        self.image_backup_suffix = ".orig"

        self._load_dataset()
        self._load_session()

        self.root = tk.Tk()
        self.root.title("Dataset Reviewer")
        self.root.geometry("1200x900")

        self.info_var = tk.StringVar(value="")
        self.path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")

        self._build_ui()
        self._jump_to_first_unreviewed()
        self._render_current()

    def _load_dataset(self) -> None:
        if not self.dataset_csv.exists():
            raise FileNotFoundError(f"dataset CSV not found: {self.dataset_csv}")

        with self.dataset_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.fieldnames = list(reader.fieldnames or [])
            self.rows = list(reader)

        if self.order == "score_asc":
            if "score" in self.fieldnames:
                def _score_key(row: Dict[str, str]) -> float:
                    raw = (row.get("score") or "").strip()
                    try:
                        return float(raw)
                    except Exception:
                        return float("inf")

                self.rows.sort(key=_score_key)
        elif self.order == "gt_len_desc":
            def _gt_len_key(row: Dict[str, str]) -> int:
                return len(row.get("gt_text") or "")

            self.rows.sort(key=_gt_len_key, reverse=True)

        if not self.rows:
            raise RuntimeError("Dataset CSV has no rows")
        if "crop_rel" not in self.fieldnames:
            raise RuntimeError("Dataset CSV must contain 'crop_rel' column")
        if "gt_text" not in self.fieldnames:
            raise RuntimeError("Dataset CSV must contain 'gt_text' column")

    def _load_session(self) -> None:
        if not self.session_json.exists():
            return
        try:
            payload = json.loads(self.session_json.read_text(encoding="utf-8"))
        except Exception:
            return

        if str(payload.get("dataset_csv", "")) != str(self.dataset_csv):
            return

        raw_decisions = payload.get("decisions", {})
        loaded: Dict[int, bool] = {}
        for k, v in raw_decisions.items():
            try:
                idx = int(k)
                loaded[idx] = bool(v)
            except Exception:
                continue

        self.decisions = {i: d for i, d in loaded.items() if 0 <= i < len(self.rows)}
        self.pos = int(payload.get("pos", 0)) if isinstance(payload.get("pos", 0), int) else 0
        self.pos = max(0, min(self.pos, len(self.rows) - 1))

    def _save_session(self) -> None:
        payload = {
            "dataset_csv": str(self.dataset_csv),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "pos": self.pos,
            "decisions": {str(k): v for k, v in self.decisions.items()},
        }
        self.session_json.parent.mkdir(parents=True, exist_ok=True)
        self.session_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(top, textvariable=self.info_var, anchor="w", font=("Arial", 12, "bold")).pack(fill=tk.X)
        tk.Label(top, textvariable=self.path_var, anchor="w", fg="#333").pack(fill=tk.X)

        img_frame = tk.Frame(self.root)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.img_label = tk.Label(img_frame, bg="#f0f0f0")
        self.img_label.pack(fill=tk.BOTH, expand=True)

        text_frame = tk.Frame(self.root)
        text_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=8)

        tk.Label(text_frame, text="GT text:", anchor="w", font=("Arial", 10, "bold")).pack(fill=tk.X)
        self.gt_text = tk.Text(text_frame, height=4, wrap=tk.WORD)
        self.gt_text.pack(fill=tk.X)

        tk.Label(text_frame, text="OCR text:", anchor="w", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(6, 0))
        self.ocr_text = tk.Text(text_frame, height=3, wrap=tk.WORD)
        self.ocr_text.pack(fill=tk.X)
        self.ocr_text.configure(state=tk.DISABLED)

        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=10, pady=8)

        tk.Button(controls, text="Keep [K]", width=14, command=self.keep_current).pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="Remove [R]", width=14, command=self.remove_current).pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="Undo [U]", width=14, command=self.undo_last).pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="Apply text [T]", width=14, command=self.apply_gt_edit).pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="Crop image [C]", width=14, command=self.crop_current_image).pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="Paste image [V]", width=14, command=self.replace_from_clipboard).pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="Load image [F]", width=14, command=self.replace_from_file).pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="Save + Quit [S]", width=16, command=self.save_and_quit).pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="Quit", width=10, command=self.quit_without_save).pack(side=tk.RIGHT, padx=3)

        tk.Label(self.root, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=10, pady=(0, 8))

        self.root.bind("<Key-k>", lambda _e: self.keep_current())
        self.root.bind("<Key-r>", lambda _e: self.remove_current())
        self.root.bind("<Key-u>", lambda _e: self.undo_last())
        self.root.bind("<Key-t>", lambda _e: self.apply_gt_edit())
        self.root.bind("<Key-c>", lambda _e: self.crop_current_image())
        self.root.bind("<Key-v>", lambda _e: self.replace_from_clipboard())
        self.root.bind("<Control-v>", lambda _e: self.replace_from_clipboard())
        self.root.bind("<Key-f>", lambda _e: self.replace_from_file())
        self.root.bind("<Key-s>", lambda _e: self.save_and_quit())

    def _current_row(self) -> Dict[str, str]:
        return self.rows[self.pos]

    def _image_path(self, row: Dict[str, str]) -> Path:
        rel = row.get("crop_rel", "")
        p = Path(rel)
        return p if p.is_absolute() else self.image_root / p

    def _set_textbox(self, widget: tk.Text, value: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        widget.configure(state=tk.DISABLED)

    def _set_gt_textbox(self, value: str) -> None:
        self.gt_text.delete("1.0", tk.END)
        self.gt_text.insert("1.0", value)

    def _commit_current_gt_text(self) -> None:
        current = self._current_row()
        edited = self.gt_text.get("1.0", tk.END).rstrip("\n")
        current["gt_text"] = edited

    def apply_gt_edit(self) -> None:
        self._commit_current_gt_text()
        self.status_var.set("Updated GT text for current sample")

    def _stats(self) -> Tuple[int, int, int]:
        kept = sum(1 for d in self.decisions.values() if d)
        removed = sum(1 for d in self.decisions.values() if not d)
        reviewed = kept + removed
        return kept, removed, reviewed

    def _current_image_path(self) -> Path:
        return self._image_path(self._current_row())

    def _backup_image_if_needed(self, image_path: Path) -> None:
        backup_path = image_path.with_suffix(image_path.suffix + self.image_backup_suffix)
        if backup_path.exists():
            return
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, backup_path)

    def _overwrite_current_image(self, new_image: Image.Image) -> None:
        image_path = self._current_image_path()
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if image_path.exists():
            self._backup_image_if_needed(image_path)
        new_image.convert("RGB").save(image_path)

    def crop_current_image(self) -> None:
        image_path = self._current_image_path()
        if not image_path.exists():
            self.status_var.set("Cannot crop: image file does not exist")
            return

        try:
            original = Image.open(image_path).convert("RGB")
        except Exception as e:
            self.status_var.set(f"Cannot open image for crop: {e}")
            return

        dlg = CropDialog(self.root, original, max_width=1400, max_height=900)
        if dlg.result_bbox is None:
            self.status_var.set("Crop cancelled")
            return

        x1, y1, x2, y2 = dlg.result_bbox
        if x2 <= x1 or y2 <= y1:
            self.status_var.set("Crop ignored: invalid selection")
            return

        cropped = original.crop((x1, y1, x2, y2))
        self._overwrite_current_image(cropped)
        self._render_current()
        self.status_var.set(f"Cropped current image to bbox ({x1}, {y1}, {x2}, {y2})")

    def replace_from_file(self) -> None:
        src = filedialog.askopenfilename(
            title="Choose replacement image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not src:
            self.status_var.set("Replace cancelled")
            return

        try:
            img = Image.open(src).convert("RGB")
        except Exception as e:
            self.status_var.set(f"Failed to load replacement image: {e}")
            return

        self._overwrite_current_image(img)
        self._render_current()
        self.status_var.set(f"Replaced current image from file: {src}")

    def replace_from_clipboard(self) -> None:
        image_obj: Optional[Image.Image] = None
        source_desc = "clipboard"

        if ImageGrab is not None:
            try:
                grabbed = ImageGrab.grabclipboard()
            except Exception:
                grabbed = None

            if isinstance(grabbed, Image.Image):
                image_obj = grabbed.convert("RGB")
            elif isinstance(grabbed, list):
                for item in grabbed:
                    p = Path(str(item))
                    if p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
                        try:
                            image_obj = Image.open(p).convert("RGB")
                            source_desc = str(p)
                            break
                        except Exception:
                            continue

        if image_obj is None:
            try:
                txt = self.root.clipboard_get().strip()
                p = Path(txt)
                if p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
                    image_obj = Image.open(p).convert("RGB")
                    source_desc = str(p)
            except Exception:
                pass

        if image_obj is None:
            self.status_var.set("Clipboard has no image. Copy an image or image-file path, then press V.")
            return

        self._overwrite_current_image(image_obj)
        self._render_current()
        self.status_var.set(f"Replaced current image from {source_desc}")

    def _render_current(self) -> None:
        row = self._current_row()
        image_path = self._image_path(row)

        decision = self.decisions.get(self.pos)
        decision_str = "UNREVIEWED"
        if decision is True:
            decision_str = "KEEP"
        elif decision is False:
            decision_str = "REMOVE"

        kept, removed, reviewed = self._stats()
        self.info_var.set(
            f"Sample {self.pos + 1}/{len(self.rows)} | Reviewed: {reviewed} | Keep: {kept} | Remove: {removed} | Current: {decision_str}"
        )
        self.path_var.set(f"Image: {image_path}")

        self._set_gt_textbox(row.get("gt_text", ""))
        self._set_textbox(self.ocr_text, row.get("ocr_text", ""))

        if image_path.exists():
            try:
                im = Image.open(image_path).convert("RGB")
                im.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)
                self.tk_img = ImageTk.PhotoImage(im)
                self.img_label.configure(image=self.tk_img, text="")
            except Exception as e:
                self.tk_img = None
                self.img_label.configure(image="", text=f"Failed to open image:\n{e}", justify=tk.LEFT)
        else:
            self.tk_img = None
            self.img_label.configure(image="", text="Image file not found", justify=tk.LEFT)

        self.status_var.set("K=Keep | R=Remove | U=Undo | T=Apply text | C=Crop | V=Paste image | F=Load image | S=Save+Quit")

    def _jump_to_first_unreviewed(self) -> None:
        for i in range(len(self.rows)):
            if i not in self.decisions:
                self.pos = i
                return
        self.pos = 0

    def _goto_next(self) -> None:
        if self.pos + 1 < len(self.rows):
            self.pos += 1
            self._render_current()
            return

        self.status_var.set("Reached last sample. Press S to save, or U to undo.")
        self._render_current()

    def _record_action(self, keep: bool) -> None:
        self._commit_current_gt_text()
        prev = self.decisions.get(self.pos)
        self.history.append(Action(row_index=self.pos, prev_decision=prev, prev_pos=self.pos))
        self.decisions[self.pos] = keep
        self._save_session()
        self._goto_next()

    def keep_current(self) -> None:
        self._record_action(True)

    def remove_current(self) -> None:
        self._record_action(False)

    def undo_last(self) -> None:
        if not self.history:
            self.status_var.set("Nothing to undo")
            return

        action = self.history.pop()
        if action.prev_decision is None:
            self.decisions.pop(action.row_index, None)
        else:
            self.decisions[action.row_index] = action.prev_decision

        self.pos = action.prev_pos
        self._save_session()
        self._render_current()
        self.status_var.set("Undid last action")

    def _write_outputs(self) -> None:
        self._commit_current_gt_text()
        kept_rows: List[Dict[str, str]] = []
        removed_rows: List[Dict[str, str]] = []

        for i, row in enumerate(self.rows):
            decision = self.decisions.get(i)
            if decision is False:
                removed_rows.append(row)
            else:
                kept_rows.append(row)

        if self.inplace:
            backup = self.dataset_csv.with_suffix(self.dataset_csv.suffix + ".bak")
            self.dataset_csv.replace(backup)
            target = self.dataset_csv
        else:
            target = self.output_csv

        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)

        self.removed_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.removed_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(removed_rows)

        self._save_session()

    def save_and_quit(self) -> None:
        self._write_outputs()
        kept, removed, _ = self._stats()
        messagebox.showinfo(
            "Saved",
            f"Saved review results.\nKeep: {kept}\nRemove: {removed}\nSession: {self.session_json}",
        )
        self.root.destroy()

    def quit_without_save(self) -> None:
        self._save_session()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Review dataset.csv samples with image + label and keep/remove/undo controls.")
    ap.add_argument("--dataset_csv", type=str, default="data/dataset.csv", help="Path to input dataset CSV")
    ap.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Root path for crop_rel images. Default: parent folder of dataset_csv",
    )
    ap.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV for kept rows. Default: <dataset>.reviewed.csv",
    )
    ap.add_argument(
        "--removed_csv",
        type=str,
        default=None,
        help="Output CSV containing removed rows. Default: <dataset>.removed.csv",
    )
    ap.add_argument(
        "--session_json",
        type=str,
        default=None,
        help="Session state path (for resume/undo history). Default: <dataset>.review_session.json",
    )
    ap.add_argument("--inplace", action="store_true", help="Overwrite input dataset_csv (creates .bak backup)")
    ap.add_argument("--max_width", type=int, default=1100, help="Max image display width")
    ap.add_argument("--max_height", type=int, default=550, help="Max image display height")
    ap.add_argument(
        "--order",
        type=str,
        default="score_asc",
        choices=["score_asc", "gt_len_desc"],
        help="Sample order: score_asc (lowest score first) or gt_len_desc (longest GT first)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    dataset_csv = Path(args.dataset_csv).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else dataset_csv.parent

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
    else:
        output_csv = dataset_csv.with_name(dataset_csv.stem + ".reviewed.csv")

    if args.removed_csv:
        removed_csv = Path(args.removed_csv).expanduser().resolve()
    else:
        removed_csv = dataset_csv.with_name(dataset_csv.stem + ".removed.csv")

    if args.session_json:
        session_json = Path(args.session_json).expanduser().resolve()
    else:
        session_json = dataset_csv.with_name(dataset_csv.stem + ".review_session.json")

    app = DatasetReviewer(
        dataset_csv=dataset_csv,
        image_root=image_root,
        output_csv=output_csv,
        removed_csv=removed_csv,
        session_json=session_json,
        inplace=args.inplace,
        max_width=args.max_width,
        max_height=args.max_height,
        order=args.order,
    )
    app.run()


if __name__ == "__main__":
    main()
