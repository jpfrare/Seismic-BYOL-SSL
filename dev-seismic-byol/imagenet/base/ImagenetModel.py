import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import timm
from timm.data import Mixup
from timm.loss import BinaryCrossEntropy
from torchmetrics import Metric

class ImagenetModel(L.LightningModule):
    def __init__(self, 
    num_classes: int, 
    optimizer, 
    optimizer_kwargs: dict, 
    lr_scheduler, 
    lr_scheduler_kwargs: dict,
    train_loss_fn,
    train_metrics: dict,
    val_loss_fn,
    val_metrics: dict,
    backbone,
    num_gpus,
    fc,
    batch_level_transforms= None):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'fc', 'train_metrics', 'val_metrics', 'train_loss_fn', 'val_loss_fn', 'batch_level_transforms'])
        self.num_classes = num_classes

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        
        self.batch_level_transforms = batch_level_transforms
        self.train_loss_fn = train_loss_fn
        self.train_metrics= nn.ModuleDict(train_metrics)

        self.val_loss_fn = val_loss_fn
        self.val_metrics= nn.ModuleDict(val_metrics)

        self.backbone = backbone
        self.fc = fc

        if isinstance(num_gpus, list) and len(num_gpus) > 1:
            self.sync_dist = True
        elif isinstance(num_gpus, int) and num_gpus > 1:
            self.sync_dist = True
        else:
            self.sync_dist = False


    def forward(self, x):
        return self.fc(self.backbone(x))
    
    def training_step(self, batch, batch_idx):
        x,y = batch

        y_clone = y.clone() if len(self.train_metrics) > 0 else y

        if self.batch_level_transforms is not None:
            x,y = self.batch_level_transforms(x,y)
        
        y_hat = self(x)

        for metric_name in self.train_metrics.keys():
            self.train_metrics[metric_name].update(y_hat, y_clone)
            self.log(f'train_{metric_name}', self.train_metrics[metric_name], on_epoch= True, prog_bar= True)

        loss = self.train_loss_fn(y_hat, y)
        self.log('train_loss', loss, on_epoch= True, prog_bar= True, sync_dist= self.sync_dist)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        y_loss = y

        for metric_name in self.val_metrics.keys():
            self.val_metrics[metric_name].update(y_hat, y)
            self.log(f'{metric_name}', self.val_metrics[metric_name], on_epoch= True, prog_bar= True)

        if isinstance(self.val_loss_fn, BinaryCrossEntropy):
            y_loss = F.one_hot(y, num_classes= self.num_classes).to(y_hat.dtype)

        val_loss = self.val_loss_fn(y_hat, y_loss)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist= self.sync_dist)
        return val_loss
    
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step', # Atualiza a cada batch, não a cada época
                    'frequency': 1,
                },
            }
        
        return optimizer
