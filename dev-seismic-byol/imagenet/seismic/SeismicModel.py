from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone, DeepLabV3, DeepLabV3PredictionHead
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR
import torch

class SeismicModel(DeepLabV3):
    def configure_optimizers(self):
        self._set_trainable_params()

        optimizer = self.optimizer(
            self.parameters(),
            **self.optimizer_kwargs
        )

        if self.lr_scheduler is None:
            return optimizer

        scheduler_kwargs = dict(self.lr_scheduler_kwargs)

        if self.lr_scheduler is OneCycleLR:
            scheduler_kwargs["total_steps"] = self.trainer.estimated_stepping_batches

        scheduler = self.lr_scheduler(
            optimizer,
            **scheduler_kwargs
        )

        if isinstance(scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        if isinstance(scheduler, OneCycleLR):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        if isinstance(scheduler, CosineAnnealingLR):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
        

    def test_and_evaluate_IoU(self, val_dataloader, metric_collection):

        self.cuda()
        self.eval()
        metric_collection.reset()
        metric_collection = metric_collection.to(self.device)

        with torch.no_grad():

            for batch_idx , (x,y) in enumerate(val_dataloader):
                print(f'Batch: {batch_idx}')

                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self(x)
                y = y.squeeze(1)

                metric_collection.update(y_hat, y)


        return metric_collection.compute()