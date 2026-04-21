from minerva.models.nets.base import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ImagenetModel(SimpleSupervisedModel):
    def configure_optimizers(self):
        # Chama o setup de parâmetros (congelamento/descongelamento)
        self._set_trainable_params()
        
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        
        if self.lr_scheduler is None:
            return optimizer

        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)

        # Se for Plateau, precisamos do dicionário de monitoramento
        if isinstance(scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss", # Métrica logada no _single_step
                    "interval": "step",    # Checa a cada validação (step do scheduler)
                    "frequency": 1,
                },
            }
        
        return [optimizer], [scheduler]