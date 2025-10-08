from pytorch_lightning.callbacks import Callback

class UnfreezeBackboneCallback(Callback):
    def __init__(self, unfreeze_at_epoch: int = 10):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_at_epoch:
            print(f"[Callback] Unfreezing backbone at epoch {trainer.current_epoch}")

            pl_module.freeze_backbone = False

            for param in pl_module.backbone.parameters():
                param.requires_grad = True
