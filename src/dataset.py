import re
import csv
import math
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CsvLineDataset(Dataset):

    def __init__(self, rows, opt, image_root):
        self.opt = opt
        self.image_root = Path(image_root).expanduser().resolve()
        self.samples = []
        out_of_char = f'[^{self.opt.character}]'

        for row in rows:
            raw_label = (row.get('gt_text') or '').strip()
            if not self.opt.sensitive:
                raw_label = raw_label.lower()
            label = re.sub(out_of_char, '', raw_label)

            if not self.opt.data_filtering_off:
                if len(label) == 0:
                    continue
                if len(label) > self.opt.batch_max_length:
                    continue

            crop_rel = (row.get('crop_rel') or '').strip()
            if not crop_rel:
                continue

            image_path = self.image_root / crop_rel
            if not image_path.exists() and crop_rel.startswith('images/'):
                image_path = self.image_root / crop_rel[len('images/'):]

            self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        if self.opt.rgb:
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('L')

        return image, label


def _parse_index_string(index_string):
    if not index_string:
        return set()
    values = set()
    for token in index_string.split(','):
        t = token.strip()
        if not t:
            continue
        values.add(int(t))
    return values


def _read_dataset_csv(dataset_csv):
    rows = []
    with open(dataset_csv, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get('crop_rel') and row.get('gt_text') is not None and row.get('pdf_id'):
                rows.append(row)
    return rows


def create_csv_split_datasets(dataset_csv, image_root, opt, val_pdf_indices='', test_pdf_indices=''):
    rows = _read_dataset_csv(dataset_csv)
    if not rows:
        raise ValueError(f'No valid rows found in CSV: {dataset_csv}')

    pdf_ids = sorted({row['pdf_id'] for row in rows})
    val_idx_set = _parse_index_string(val_pdf_indices)
    test_idx_set = _parse_index_string(test_pdf_indices)

    overlap = val_idx_set.intersection(test_idx_set)
    if overlap:
        raise ValueError(f'Validation and test indices overlap: {sorted(overlap)}')

    max_idx = len(pdf_ids) - 1
    for idx in sorted(val_idx_set.union(test_idx_set)):
        if idx < 0 or idx > max_idx:
            raise ValueError(f'PDF index out of range: {idx}. Valid range is 0..{max_idx}')

    val_pdf_ids = {pdf_ids[idx] for idx in val_idx_set}
    test_pdf_ids = {pdf_ids[idx] for idx in test_idx_set}

    train_rows = []
    val_rows = []
    test_rows = []

    for row in rows:
        pdf_id = row['pdf_id']
        if pdf_id in val_pdf_ids:
            val_rows.append(row)
        elif pdf_id in test_pdf_ids:
            test_rows.append(row)
        else:
            train_rows.append(row)

    train_dataset = CsvLineDataset(train_rows, opt=opt, image_root=image_root)
    val_dataset = CsvLineDataset(val_rows, opt=opt, image_root=image_root)
    test_dataset = CsvLineDataset(test_rows, opt=opt, image_root=image_root)

    split_info = {
        'pdf_ids': pdf_ids,
        'val_pdf_ids': sorted(val_pdf_ids),
        'test_pdf_ids': sorted(test_pdf_ids),
        'train_count': len(train_dataset),
        'val_count': len(val_dataset),
        'test_count': len(test_dataset),
    }

    return train_dataset, val_dataset, test_dataset, split_info




class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(1.0)  # fill with white
        Pad_img[:, :, :w] = img  # right pad
        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32):
        self.imgH = imgH

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)
        # Dynamically determine max width for this batch
        widths = []
        resized_images = []
        input_channel = 3 if images[0].mode == 'RGB' else 1
        for image in images:
            w, h = image.size
            ratio = w / float(h)
            resized_w = math.ceil(self.imgH * ratio)
            widths.append(resized_w)
        max_w = max(widths)
        transform = NormalizePAD((input_channel, self.imgH, max_w))
        for image in images:
            w, h = image.size
            ratio = w / float(h)
            resized_w = math.ceil(self.imgH * ratio)
            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_images.append(transform(resized_image))
        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        return image_tensors, labels
