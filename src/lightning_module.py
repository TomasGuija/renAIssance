from __future__ import annotations
from types import SimpleNamespace
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from model import Model
from utils import CTCLabelConverter


class OCRLightningCTCModule(pl.LightningModule):
    def __init__(
        self,
        character: str,                   # Could be empty, automatically linked to character in the data configuration. 
        batch_max_length: int = 25,       # Linked to data configuration
        imgH: int = 32,                   # Linked to data configuration
        rgb: bool = False,                # Linked to data configuration
        sensitive: bool = False,          # Linked to data configuration
        data_filtering_off: bool = False, # Linked to data configuration
        output_channel: int = 512,
        hidden_size: int = 256,
        lr: float = 1e-4,
        pretrained_path: str = '',
        wandb_log_every_n_epochs: int = 0,
        wandb_num_logs: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.opt = SimpleNamespace(
            character=character,
            batch_max_length=batch_max_length,
            imgH=imgH,
            rgb=rgb,
            sensitive=sensitive,
            data_filtering_off=data_filtering_off,
            input_channel=1 if not rgb else 3,
            output_channel=output_channel,
            hidden_size=hidden_size,
            lr=lr,
        )

        self.pretrained_path = pretrained_path
        self._pretrained_loaded = False
        self.wandb_log_every_n_epochs = wandb_log_every_n_epochs
        self.wandb_num_logs = wandb_num_logs
        self._train_examples = None
        self._val_examples = None

        self.converter = CTCLabelConverter(self.opt.character)
        self.opt.num_class = len(self.converter.character)
        self.model = Model(self.opt)
        self.criterion = torch.nn.CTCLoss(zero_infinity=True)

    def on_fit_start(self):
        if self.pretrained_path and not self._pretrained_loaded:
            state_dict = torch.load(self.pretrained_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            self._pretrained_loaded = True
            self.print(f'Loaded pretrained weights from: {self.pretrained_path} (strict=False)')

    def forward(self, image):
        return self.model(image)

    def _ctc_loss_and_decode(self, image, labels):
        text_for_loss, length_for_loss = self.converter.encode(labels, batch_max_length=self.opt.batch_max_length)
        preds = self.model(image)
        preds_size = torch.full(
            size=(image.size(0),),
            fill_value=preds.size(1),
            dtype=torch.long,
            device=image.device,
        )
        loss = self.criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index.detach(), preds_size.detach())
        return loss, preds_str

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

    def _build_wandb_examples(self, wandb, examples_dict):
        images = examples_dict['images']
        labels = examples_dict['labels']
        preds = examples_dict['preds']
        num_items = min(self.wandb_num_logs, len(labels), images.size(0))

        out = []
        for i in range(num_items):
            img = self._tensor_to_wandb_image(images[i])
            caption = f"gt: {labels[i]}\npred: {preds[i]}"
            out.append(wandb.Image(img, caption=caption))
        return out

    def _log_examples_to_wandb(self, split: str, examples_dict):
        if examples_dict is None:
            return

        logger = self._pick_wandb_logger()
        if logger is None:
            return

        try:
            import wandb
        except Exception:
            return

        examples = self._build_wandb_examples(wandb, examples_dict)
        if not examples:
            return

        run = logger.experiment
        epoch_step = self.current_epoch + 1
        run.log(
            {
                f'{split}/examples': examples,
                'epoch': epoch_step,
            },
        )

    def on_train_epoch_end(self):
        if not self._should_log_wandb_examples():
            self._train_examples = None
            return

        self._log_examples_to_wandb('train', self._train_examples)
        self._train_examples = None

    def training_step(self, batch, batch_idx):
        image_tensors, labels = batch
        images = image_tensors.to(self.device)

        loss, preds_str = self._ctc_loss_and_decode(images, labels)

        if self._should_log_wandb_examples() and batch_idx == 0:
            self._train_examples = {
                'images': image_tensors.detach().cpu(),
                'labels': list(labels),
                'preds': list(preds_str),
            }

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        return loss
    
    def on_validation_epoch_end(self):
        if not self._should_log_wandb_examples():
            self._val_examples = None
            return

        self._log_examples_to_wandb('val', self._val_examples)
        self._val_examples = None

    def validation_step(self, batch, batch_idx):
        image_tensors, labels = batch
        images = image_tensors.to(self.device)

        loss, preds_str = self._ctc_loss_and_decode(images, labels)

        if self._should_log_wandb_examples() and batch_idx == 0:
            self._val_examples = {
                'images': image_tensors.detach().cpu(),
                'labels': list(labels),
                'preds': list(preds_str),
            }

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))

    def configure_optimizers(self):
        filtered_parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(filtered_parameters, lr=self.opt.lr, betas=(0.9, 0.999))
        return optimizer
