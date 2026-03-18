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

You may find the pipeline for using a pretrained model 

