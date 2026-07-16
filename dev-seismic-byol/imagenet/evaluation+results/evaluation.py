from metrics import Metrics
from modelInfo import ModelInfo
from pathlib import Path

ROOT = '/petrobr/parceirosbr/spfm/joao.frare/logs+checkpoints_imagenet'
SAVE_FINETUNE_ROOT = Path('data/finetune')
SAVE_PRETRAIN_ROOT = Path('data/pretrain')
full_dataset_model = ModelInfo(root_path = Path(ROOT), reduction_mode = 'full')
metrics = Metrics([full_dataset_model])

metrics.show_best_pretrain_metrics('ModelsPretrainBestStep', SAVE_PRETRAIN_ROOT)
metrics.group_plot(
        title= 'Acc1 x Steps',
        save_path= SAVE_PRETRAIN_ROOT,
        y_axis= 'acc1',
        x_axis= 'step',
        mul_factor = 100)
metrics.individual_plot(
        title= 'Train and Val x Steps',
        save_path= SAVE_PRETRAIN_ROOT,
        y_axis= ['val_loss','train_loss'],
        x_axis= 'step',
        yscale= 'log',
        ncols= 1)


for dataset in ['f3_N', 'seam_ai_N']:
    for protocol in ['full_finetuning_deeplab', 'full_freeze_linear']:
        metrics.save_miou_table(f'miou_table_{dataset}_{protocol}', SAVE_FINETUNE_ROOT, dataset, protocol)

        metrics.individual_plot(
        title= f'Val and Train x epoch - {dataset}|{protocol}',
        save_path= SAVE_FINETUNE_ROOT,
        y_axis= ['val_loss', 'train_loss'],
        x_axis= 'epoch',
        yscale= 'log',
        ncols= 1,
        finetune= True,
        dataset= dataset,
        protocol= protocol)