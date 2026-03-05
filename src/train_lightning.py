from __future__ import annotations

import logging

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.cli import LightningCLI

from lightning_data import OCRDataModule
from lightning_module import OCRLightningCTCModule


def _to_plain(obj):
    basic = (str, int, float, bool, type(None))
    if isinstance(obj, basic):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    try:
        data = vars(obj)
    except TypeError:
        try:
            return str(obj)
        except Exception:
            return repr(obj)
    return {k: _to_plain(v) for k, v in data.items()}


def _log_config(cli: 'LightningCLI'):
    logging.info('Loaded configuration:')
    cfg_ns = getattr(cli.config, 'fit', cli.config)
    cfg_dict = _to_plain(cfg_ns)
    logging.info('\n' + yaml.safe_dump(cfg_dict, sort_keys=False))


class OCRLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.character', 'model.character')
        parser.link_arguments('data.batch_max_length', 'model.batch_max_length')
        parser.link_arguments('data.imgH', 'model.imgH')
        parser.link_arguments('data.imgW', 'model.imgW')
        parser.link_arguments('data.rgb', 'model.rgb')
        parser.link_arguments('data.PAD', 'model.PAD')
        parser.link_arguments('data.sensitive', 'model.sensitive')
        parser.link_arguments('data.data_filtering_off', 'model.data_filtering_off')

    def before_fit(self):
        _log_config(self)


def main():
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(333, workers=True)
    OCRLightningCLI(
        model_class=OCRLightningCTCModule,
        datamodule_class=OCRDataModule,
        save_config_kwargs={'overwrite': True},
        seed_everything_default=333,
    )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
