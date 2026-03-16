# Dataset Construction Pipeline

This repository contains the scripts used to construct the OCR training dataset used in this project. The dataset is built from historical PDF documents and their corresponding transcriptions stored in `.docx` files.

The process combines automated extraction (segmentation, OCR, alignment) with manual verification and filtering to ensure high-quality labels.

---

# Overview of the Pipeline

The dataset is constructed in the following stages:

1. Render PDF pages into images  
2. Automatically construct a preliminary dataset using segmentation and OCR  
3. Preprocess the extracted line images  
4. Manually review samples and remove incorrect ones  
5. Remove incomplete annotations  
6. Normalize punctuation  
7. Remove incorrectly segmented multi-line samples  
8. Filter the dataset to keep only Spanish sentences

---

# 1. Render PDF Pages

The first step converts PDF documents into raster images so that line segmentation and OCR models can operate on them.

Script used:

```
scripts/render_pdf.py
```

Each page of each PDF is rendered as a PNG image using **PyMuPDF**.

Example output structure:

```
data/pages/
  document_1/
    1.png
    2.png
    3.png
  document_2/
    1.png
    2.png
```

Pages are rendered at high resolution (default **400 DPI**) to preserve text quality.

---

# 2. Automatic Dataset Construction

Once the page images are generated, an automatic pipeline constructs an initial dataset.

Script used:

```
scripts/construct_dataset.py
```

This script performs the following steps:

### Line segmentation

Each page image is segmented into individual text lines using **Kraken**.

### Line cropping

Detected lines are cropped from the page image and saved as individual images.

### Initial OCR prediction

Each cropped line is processed with **Tesseract OCR** to obtain a preliminary transcription.

### Ground-truth extraction

The corresponding transcription is extracted from `.docx` files that contain the annotated text of the document.

The script parses page markers such as:

```
PDF p1
PDF p2
PDF p3
```

to associate transcription lines with the correct page.

### OCR–GT alignment

OCR predictions are aligned with ground-truth lines using a **monotonic sequence alignment algorithm** based on string similarity.

This produces pairs:

```
(line_image, gt_text)
```

along with a similarity score.

### Dataset output

Accepted matches are written to:

```
dataset.csv
```

Each row contains metadata such as:

- PDF identifier
- page number
- cropped image path
- ground-truth text
- OCR text
- similarity score
- bounding box coordinates

Unassigned OCR lines or unmatched ground-truth lines are stored separately for inspection.

---

# 3. Image Preprocessing

After dataset construction, all cropped line images are preprocessed to standardize their appearance.

Script used:

```
scripts/preprocess_images.py
```

Preprocessing steps include:

- conversion to grayscale
- contrast normalization using percentile clipping
- optional gamma correction
- resizing to a fixed height (default **32 pixels**)

This ensures the images have consistent contrast and scale before training.

---

# 4. Manual Dataset Review

Even with automatic alignment, some samples contain incorrect labels or segmentation errors.

To correct these cases, a manual review tool is used.

Script used:

```
scripts/review_dataset.py
```

This script launches a graphical interface where each sample can be inspected.

For each line image, the reviewer can:

- keep the sample
- remove the sample
- edit the ground-truth text
- crop the image if segmentation is slightly incorrect
- replace the image if necessary

Samples with incorrect labels are removed from the dataset.

---

# 5. Remove Incomplete Annotations

Some samples contain ellipses such as:

```
...
…
```

These typically correspond to incomplete or uncertain annotations in the source transcription.

All samples containing these markers are removed from the dataset.

---

# 6. Normalize Punctuation

Different hyphen characters are normalized so that punctuation is consistent across the dataset.

For example:

```
–  →  -
```

This prevents the model from learning inconsistent punctuation variants.

---

# 7. Remove Incorrectly Segmented Multi-Line Samples

Some samples correspond to **two visual lines merged into a single transcription**.

This occurs when the original word processor automatically wrapped text to the next line but did not insert an explicit newline character.

These cases are typically detected by examining **very long labels** and manually removing them.

---

# 8. Language Filtering

Finally, the dataset is filtered to keep only **Spanish sentences**.

Script used:

```
scripts/keep_spanish.py
```

Language detection is performed using a pretrained **FastText language identification model (`lid.176.bin`)**.

The script predicts the language of each ground-truth string and keeps only rows where the predicted language is Spanish (`es`).

---

# Final Dataset

After all processing and filtering steps, the resulting dataset contains:

- cropped line images
- verified ground-truth transcriptions
- normalized punctuation
- Spanish text only
- manually validated samples

The final dataset can then be used to train OCR models on historical Spanish documents.