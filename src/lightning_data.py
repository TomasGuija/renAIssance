from __future__ import annotations

from types import SimpleNamespace

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from dataset import AlignCollate, create_csv_split_datasets


class OCRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_csv: str,
        image_root: str,
        val_pdf_indices: str = '',
        test_pdf_indices: str = '',
        workers: int = 4,
        batch_size: int = 64,
        batch_max_length: int = 25,
        imgH: int = 32,
        rgb: bool = False,
        sensitive: bool = False,
        data_filtering_off: bool = False,
        character: str = '0123456789abcdefghijklmnopqrstuvwxyz',
    ):
        super().__init__()
        self.save_hyperparameters()
        self.opt = SimpleNamespace(
            dataset_csv=dataset_csv,
            image_root=image_root,
            val_pdf_indices=val_pdf_indices,
            test_pdf_indices=test_pdf_indices,
            workers=workers,
            batch_size=batch_size,
            batch_max_length=batch_max_length,
            imgH=imgH,
            rgb=rgb,
            sensitive=sensitive,
            data_filtering_off=data_filtering_off,
            character=character,
        )
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.split_info = None

    def setup(self, stage=None):
        train_dataset, val_dataset, test_dataset, split_info = create_csv_split_datasets(
            dataset_csv=self.opt.dataset_csv,
            image_root=self.opt.image_root,
            opt=self.opt,
            val_pdf_indices=self.opt.val_pdf_indices,
            test_pdf_indices=self.opt.test_pdf_indices,
        )

        if len(train_dataset) == 0:
            raise ValueError('Training split is empty. Adjust val/test pdf indices.')
        if len(val_dataset) == 0:
            raise ValueError('Validation split is empty. Adjust val_pdf_indices.')

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.split_info = split_info

    def _collate(self):
        return AlignCollate(imgH=self.opt.imgH)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=int(self.opt.workers),
            collate_fn=self._collate(),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=self._collate(),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=self._collate(),
            pin_memory=True,
        )
