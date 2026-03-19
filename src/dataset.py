import re
import csv
import math
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import v2

class CsvLineDataset(Dataset):

    def __init__(self, rows, opt, image_root, is_train=False):
        self.opt = opt
        self.image_root = Path(image_root).expanduser().resolve()
        self.samples = []
        self.is_train = is_train

        allowed_chars = re.escape(self.opt.character)
        invalid_char_pattern = re.compile(f'[^{allowed_chars}]')

        self.aug = v2.Compose([
            v2.RandomApply([
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))
            ], p=0.25),

            v2.RandomApply([
                v2.RandomAdjustSharpness(sharpness_factor=1.5)
            ], p=0.25),
        ])
        
        for row in rows:
            raw_label = (row.get('gt_text') or '').strip()
            if not self.opt.sensitive:
                raw_label = raw_label.lower()
            
            label = invalid_char_pattern.sub('', raw_label)

            if not self.opt.data_filtering_off:
                if len(label) == 0:
                    continue
                if len(label) > self.opt.batch_max_length:
                    continue

            crop_rel = (row.get('crop_rel') or '').strip()

            if not crop_rel:
                continue

            image_path = self.image_root / crop_rel

            self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        if self.opt.rgb:
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('L')
        
        if self.is_train:
            image = self.aug(image)

        return image, label


def _read_dataset_csv(dataset_csv):
    rows = []
    with open(dataset_csv, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get('crop_rel') and row.get('gt_text') is not None and row.get('pdf_id'):
                rows.append(row)
    return rows


def create_csv_split_datasets(dataset_csv, image_root, opt, val_indices=[], test_indices=[]):
    rows = _read_dataset_csv(dataset_csv)
    if not rows:
        raise ValueError(f'No valid rows found in CSV: {dataset_csv}')
    
    val_idx_set = set(val_indices)
    test_idx_set = set(test_indices)

    pdf_ids = sorted({row['pdf_id'] for row in rows})

    max_idx = len(pdf_ids) - 1
    for idx in sorted(val_idx_set | test_idx_set):
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

    train_dataset = CsvLineDataset(train_rows, opt=opt, image_root=image_root, is_train=True)
    val_dataset = CsvLineDataset(val_rows, opt=opt, image_root=image_root, is_train=False)
    test_dataset = CsvLineDataset(test_rows, opt=opt, image_root=image_root, is_train=False)

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

    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, img):
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)

        _, _, w = img.size()
        pad_img = torch.full(self.max_size, 1.0, dtype=img.dtype, device=img.device)
        pad_img[:, :, :w] = img
        return pad_img


class AlignCollate(object):

    def __init__(self, imgH=32):
        self.imgH = imgH

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        images, labels = zip(*batch)
        input_channel = 3 if images[0].mode == 'RGB' else 1

        resized_widths = []
        for image in images:
            w, h = image.size
            ratio = w / float(h)
            resized_widths.append(math.ceil(self.imgH * ratio))
            
        max_w = max(resized_widths)
        transform = NormalizePAD((input_channel, self.imgH, max_w))

        resized_images = []
        for image, resized_w in zip(images, resized_widths):
            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_images.append(transform(resized_image))

        image_tensors = torch.stack(resized_images, dim=0)
        return image_tensors, labels
