from __future__ import annotations

from types import SimpleNamespace

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger

from model import Model
from utils import CTCLabelConverter


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, replace))
        prev = cur
    return prev[-1]


class OCRLightningCTCModule(pl.LightningModule):
    def __init__(
        self,
        character: str,
        batch_max_length: int = 25,
        imgH: int = 32,
        imgW: int = 1600,
        rgb: bool = False,
        PAD: bool = True,
        sensitive: bool = False,
        data_filtering_off: bool = False,
        Transformation: str = 'None',
        FeatureExtraction: str = 'ResNet',
        SequenceModeling: str = 'BiLSTM',
        Prediction: str = 'CTC',
        num_fiducial: int = 20,
        input_channel: int = 1,
        output_channel: int = 512,
        hidden_size: int = 256,
        adam: bool = True,
        lr: float = 1e-4,
        beta1: float = 0.9,
        rho: float = 0.95,
        eps: float = 1e-8,
        pretrained_path: str = '',
        finetune: bool = True,
        wandb_log_every_n_epochs: int = 0,
        wandb_num_logs: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.opt = SimpleNamespace(
            character=character,
            batch_max_length=batch_max_length,
            imgH=imgH,
            imgW=imgW,
            rgb=rgb,
            PAD=PAD,
            sensitive=sensitive,
            data_filtering_off=data_filtering_off,
            Transformation=Transformation,
            FeatureExtraction=FeatureExtraction,
            SequenceModeling=SequenceModeling,
            Prediction=Prediction,
            num_fiducial=num_fiducial,
            input_channel=input_channel,
            output_channel=output_channel,
            hidden_size=hidden_size,
            adam=adam,
            lr=lr,
            beta1=beta1,
            rho=rho,
            eps=eps,
        )

        self.pretrained_path = pretrained_path
        self.finetune = finetune
        self._pretrained_loaded = False
        self.wandb_log_every_n_epochs = max(0, int(wandb_log_every_n_epochs))
        self.wandb_num_logs = max(1, int(wandb_num_logs))
        self._val_examples = None

        if self.opt.Prediction != 'CTC':
            raise ValueError('This Lightning module currently supports Prediction=CTC only.')

        self.converter = CTCLabelConverter(self.opt.character)
        self.opt.num_class = len(self.converter.character)
        if self.opt.rgb:
            self.opt.input_channel = 3

        self.model = Model(self.opt)
        self.criterion = torch.nn.CTCLoss(zero_infinity=True)

    def on_fit_start(self):
        if self.pretrained_path and not self._pretrained_loaded:
            state_dict = torch.load(self.pretrained_path, map_location='cpu')
            strict = not self.finetune
            self.model.load_state_dict(state_dict, strict=strict)
            self._pretrained_loaded = True
            self.print(f'Loaded pretrained weights from: {self.pretrained_path} (strict={strict})')

    def forward(self, image, text):
        return self.model(image, text)

    def _ctc_loss_and_decode(self, image, labels):
        text_for_loss, length_for_loss = self.converter.encode(labels, batch_max_length=self.opt.batch_max_length)
        text_for_pred = torch.LongTensor(image.size(0), self.opt.batch_max_length + 1).fill_(0).to(image.device)

        preds = self.model(image, text_for_pred)
        preds_size = torch.IntTensor([preds.size(1)] * image.size(0)).to(image.device)

        loss = self.criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index.data, preds_size.data)
        return loss, preds_str

    def _compute_batch_metrics(self, preds_str, labels):
        correct = 0
        norm_ed_sum = 0.0

        for gt, pred in zip(labels, preds_str):
            if self.opt.sensitive and self.opt.data_filtering_off:
                gt_cmp = gt.lower()
                pred_cmp = pred.lower()
            else:
                gt_cmp = gt
                pred_cmp = pred

            if pred_cmp == gt_cmp:
                correct += 1

            if len(gt_cmp) == 0 or len(pred_cmp) == 0:
                norm_ed = 0.0
            elif len(gt_cmp) > len(pred_cmp):
                norm_ed = 1.0 - (_levenshtein(pred_cmp, gt_cmp) / len(gt_cmp))
            else:
                norm_ed = 1.0 - (_levenshtein(pred_cmp, gt_cmp) / len(pred_cmp))
            norm_ed_sum += norm_ed

        batch_size = max(len(labels), 1)
        acc = correct / batch_size
        norm_ed_avg = norm_ed_sum / batch_size
        return acc, norm_ed_avg

    def _should_log_wandb_examples(self) -> bool:
        if self.wandb_log_every_n_epochs <= 0:
            return False
        return ((self.current_epoch + 1) % self.wandb_log_every_n_epochs) == 0

    def _pick_wandb_logger(self):
        if self.trainer is None:
            return None
        loggers = getattr(self.trainer, 'loggers', None)
        if not loggers:
            return None
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None

    @staticmethod
    def _tensor_to_wandb_image(image_tensor: torch.Tensor):
        # Input tensor is normalized to [-1, 1]. Convert to uint8 HWC for W&B.
        img = image_tensor.detach().float().cpu()
        img = (img + 1.0) * 0.5
        img = torch.clamp(img, 0.0, 1.0)
        img = (img * 255.0).to(torch.uint8)

        if img.ndim == 3 and img.shape[0] == 1:
            hw = img[0].numpy()
            return hw

        if img.ndim == 3:
            hwc = img.permute(1, 2, 0).numpy()
            return hwc

        return img.numpy()

    def training_step(self, batch, batch_idx):
        image_tensors, labels = batch
        image = image_tensors.to(self.device)

        text, length = self.converter.encode(labels, batch_max_length=self.opt.batch_max_length)
        preds = self.model(image, text)
        preds_size = torch.IntTensor([preds.size(1)] * image.size(0)).to(image.device)
        loss = self.criterion(preds.log_softmax(2).permute(1, 0, 2), text, preds_size, length)

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=image.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        image_tensors, labels = batch
        image = image_tensors.to(self.device)
        loss, preds_str = self._ctc_loss_and_decode(image, labels)
        acc, norm_ed = self._compute_batch_metrics(preds_str, labels)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=image.size(0))
        self.log('val/accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=image.size(0))
        self.log('val/norm_ed', norm_ed, on_step=False, on_epoch=True, prog_bar=True, batch_size=image.size(0))

        if self._should_log_wandb_examples() and batch_idx == 0:
            self._val_examples = {
                'images': image_tensors.detach().cpu(),
                'labels': list(labels),
                'preds': list(preds_str),
            }

    def on_validation_epoch_end(self):
        if not self._should_log_wandb_examples():
            self._val_examples = None
            return

        logger = self._pick_wandb_logger()
        if logger is None or self._val_examples is None:
            self._val_examples = None
            return

        try:
            import wandb
        except Exception:
            self._val_examples = None
            return

        images = self._val_examples['images']
        labels = self._val_examples['labels']
        preds = self._val_examples['preds']
        num_items = min(self.wandb_num_logs, len(labels), images.size(0))

        examples = []
        for i in range(num_items):
            img = self._tensor_to_wandb_image(images[i])
            caption = f"gt: {labels[i]}\npred: {preds[i]}"
            examples.append(wandb.Image(img, caption=caption))

        logger.experiment.log(
            {
                'val/examples': examples,
                'epoch': self.current_epoch + 1,
            },
            step=self.global_step,
        )

        self._val_examples = None

    def test_step(self, batch, batch_idx):
        image_tensors, labels = batch
        image = image_tensors.to(self.device)
        loss, preds_str = self._ctc_loss_and_decode(image, labels)
        acc, norm_ed = self._compute_batch_metrics(preds_str, labels)

        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=image.size(0))
        self.log('test/accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=image.size(0))
        self.log('test/norm_ed', norm_ed, on_step=False, on_epoch=True, prog_bar=True, batch_size=image.size(0))

    def configure_optimizers(self):
        filtered_parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.opt.adam:
            optimizer = torch.optim.Adam(filtered_parameters, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        else:
            optimizer = torch.optim.Adadelta(filtered_parameters, lr=self.opt.lr, rho=self.opt.rho, eps=self.opt.eps)
        return optimizer
