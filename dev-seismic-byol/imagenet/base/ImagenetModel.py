import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import timm
from timm.data import Mixup
from timm.loss import BinaryCrossEntropy
from timm.scheduler import CosineLRScheduler
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
    
    def test_step(self, batch, batch_idx):
        # Espelha exatamente o comportamento da validação
        return self.validation_step(batch, batch_idx)
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step', 
                    'frequency': 1,
                },
            }
        
        return optimizer
    
    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step_update(self.global_step)

    def evaluate_acc(self, val_loader) -> list:
        '''função que roda uma validação e retorna uma lista onde lista[i] é a acurácia referente a cada classe
        como o dataset de validação é balanceado, a média dessa lista retorna a acurácia total'''
        
        print("len(val_loader):", len(val_loader))
        print("batch size:", val_loader.batch_size)
        print("self.device:", self.device)
        print("backbone device:", next(self.backbone.parameters()).device)
        print("fc device:", next(self.fc.parameters()).device)
        class_acc = {
            c: {'count': 0, 'matches': 0}
            for c in range(self.num_classes)
        }
        self.eval()

        with torch.no_grad():

            for var, (x, y) in enumerate(val_loader):

                print(f'Batch {var}')

                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self(x)

                y_hat = torch.argmax(y_hat, dim=1)

                for target, prediction in zip(y, y_hat):

                    target = target.item()
                    prediction = prediction.item()

                    class_acc[target]['count'] += 1

                    if target == prediction:
                        class_acc[target]['matches'] += 1

        return [
            class_acc[c]['matches'] / class_acc[c]['count']
            for c in class_acc
            if class_acc[c]['count'] > 0
        ]
