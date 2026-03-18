<p align="center">
  <img src="images/logo.svg" alt="Google Summer of Code" height="">
</p>

# HumanAI – GSoC 2026 Entry Project

This repository contains a small OCR pipeline built as an entry project for the **RenAIssance initiative within HumanAI** as part of the **Google Summer of Code 2026 application process**.

The goal of the project is to explore the transcription of **historical Spanish printed texts** using a combination of **deep learning–based Optical Character Recognition (OCR)** and **Large Language Model (LLM) post-processing**.

Rather than developing a model from scratch, the focus of this work is on **building a robust baseline pipeline** using existing methods from the literature and adapting them to a new dataset of early Spanish prints.

---

# Project Goals

The main objective of this project is to **approach the historical OCR problem from a practical perspective** by building a complete working pipeline.

The project focuses on:

* Understanding the challenges of **OCR on historical Spanish prints**
* Building a **baseline OCR model using existing architectures**
* Preparing and cleaning a **dataset of historical text lines**
* Training a model capable of recognizing the text
* Implementing an **LLM-based post-processing step** to correct OCR errors using contextual information
* Creating a **simple inference workflow** capable of transcribing full pages

---

# Methodology Overview

This repository contains the code used for:

## Dataset creation

All the code for creating the training dataset can be found in the
[`scripts`](scripts/) folder, which includes a detailed [`README.md`](scripts/README.md) describing the full process.

## OCR Recognition

A neural OCR model is trained on the dataset of historical Spanish prints constructed for this project. The model is designed purely for **text transcription**, taking as input images representing **individual lines of text**.

The extraction of these line images is described in the dataset construction pipeline at [`README.md`](scripts/README.md). Although the model itself operates on line images, full-page transcription remains possible by first applying a pretrained **line segmentation model**, as demonstrated in [`inference.ipynb`](notebooks/inference.ipynb).

The OCR implementation is based on the  
[`Deep Text Recognition Benchmark`](https://github.com/clovaai/deep-text-recognition-benchmark) framework by Baek et al.

This framework provides a modular architecture for scene text recognition combining:

- CNN feature extraction
- Sequence modeling (BiLSTM)
- CTC decoding

The original codebase provides a flexible and highly modular implementation supporting multiple architectures. The code contained in [`src`](src) is a simplified version of the original repository, retaining only the components required for this project.

Additionally, the codebase was adapted to train models using **PyTorch Lightning**, which simplifies the training loop and improves experiment organization. Experiment tracking and logging are handled using [`Weights & Biases`](https://wandb.ai/site) (`wandb`).

My contributions in this project focus primarily on the **training pipeline**, **dataset preprocessing**, and **data loading infrastructure**.

## Training

The training pipeline is configured using **YAML configuration files**.  
An example configuration can be found at: [`configs/train_lightning.yaml`](configs/train_lightning.yaml). For training a model, use the command: 

```bash
python src/train_lightning.py fit -c configs/train_lightning.yaml
```

Optionally, you may use

```bash
--ckpt_path path/to/checkpoint
```

to resume training from a pretrained model. 

## Inference and LLM Post-Processing

An example inference pipeline is provided in [`inference.ipynb`](notebooks/inference.ipynb).

The notebook demonstrates the full workflow for transcribing a historical page:

1. **PDF page processing** – extracting page images from a PDF document.
2. **Line segmentation** – detecting and isolating individual text lines.
3. **OCR inference** – running a pretrained OCR checkpoint on the extracted line images.
4. **LLM post-processing** – correcting systematic OCR errors using a locally hosted language model.

The notebook is intended as a practical guide for reproducing the inference pipeline and adapting it to new documents.

---

# Results

A model was trained using the dataset constructed for this project. Due to the limited amount of available data, the dataset was split at the **document level**, using **five documents for training** and **one document for validation**.

Details about the selected hyperparameters can be found in the [`training configuration`](configs/train_lightning.yaml) and the model implementation in [`src/lightning_module.py`](src/lightning_module.py).


## Qualitative Evaluation

To qualitatively evaluate the model, see the [`inference notebook`](notebooks/inference.ipynb), which demonstrates the full transcription pipeline. The notebook performs:

1. line segmentation on page images  
2. OCR prediction for each line  
3. LLM-based post-processing  


## Quantitative Evaluation

For quantitative evaluation, I measured the **Character Error Rate (CER)** using the [`Levenshtein distance`](https://pypi.org/project/python-Levenshtein/) between predicted text and ground-truth labels.

Evaluation is performed using:

[`src/evaluate_ocr.py`](src/evaluate_ocr.py)

The evaluation was conducted on the validation document:

**Covarrubias – *Tesoro de la lengua***.

The resulting **CER was 0.1868**.

Due to the limited amount of available training data, the model exhibits a clear tendency to **overfit**. When evaluated on the training split (containing the remaining five documents), the resulting **CER is 0.0264**, which is significantly lower than the validation CER.


## Potential improvements

Exploring broader improvements in OCR architectures is outside the scope of this project, whose primary objective was to design and validate a complete OCR pipeline, covering all stages from dataset construction and preprocessing to model training, inference, and LLM-based post-processing.

However, one important component that has not yet been incorporated into the current pipeline is **data augmentation**. Due to the limited size of the available dataset and the clear overfitting observed during training, the model would likely benefit significantly from augmentation techniques designed to increase the variability of the training data.

In the context of historical document OCR, several augmentation strategies could be particularly useful, such as **geometric transformations**, **photometric augmentations** and **synthetic noise and artifacts**.

These augmentations could be incorporated directly into the existing **PyTorch data loading pipeline**, allowing transformations to be applied dynamically during training. This would increase the effective diversity of the training set without requiring additional annotated data.

---

# References

[1] Baek, J., Kim, G., Lee, J., Park, S., Han, D., Yun, S., Oh, S. J., & Lee, H. (2019).  
**What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis.**  
Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

[2] Biewald, L. (2020).  
**Experiment Tracking with Weights and Biases.**  
Software available at https://www.wandb.com/.