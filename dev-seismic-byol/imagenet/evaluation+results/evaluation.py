from metrics import Metrics
from modelInfo import ModelInfo
from pathlib import Path

ROOT = '/petrobr/parceirosbr/spfm/joao.frare/logs+checkpoints_imagenet'
SAVE_FINETUNE_ROOT = Path('data/finetune')
SAVE_PRETRAIN_ROOT = Path('data/pretrain')

configs = {
    1000: [640, 320, 160, 80, 13],
    500:  [1300, 640, 320, 160],
    250:  [1300, 640, 320],
    125:  [1300, 640],
    10:   [1300],
}

models = [ModelInfo(root_path = Path(ROOT), reduction_mode = 'full')]

for num_classes, list_per_class in configs.items():
        for per_class in list_per_class:
                models.append(ModelInfo(root_path= Path(ROOT), reduction_mode= 'default', num_classes= num_classes, per_class= per_class))


models.append(ModelInfo(root_path = Path(ROOT), scratch= True))
metrics = Metrics(models)

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
        ncols= 4)


for dataset in ['f3_N', 'seam_ai_N']:
    for protocol in ['full_finetuning_deeplab', 'full_freeze_linear']:
        metrics.save_miou_table(f'miou_table_{dataset}_{protocol}', SAVE_FINETUNE_ROOT, dataset, protocol)

        metrics.individual_plot(
                title= f'Val and Train x epoch - {dataset}|{protocol}',
                save_path= SAVE_FINETUNE_ROOT,
                y_axis= ['val_loss', 'train_loss'],
                x_axis= 'epoch',
                yscale= 'log',
                ncols= 4,
                finetune= True,
                dataset= dataset,
                protocol= protocol)

        metrics.plot_heatmap(
                title= f'HeatMap mean mIoU - {dataset}|{protocol}',
                save_path= SAVE_FINETUNE_ROOT,
                dataset= dataset,
                protocol= protocol,
        )