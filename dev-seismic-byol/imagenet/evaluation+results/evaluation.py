from metrics import Metrics
from modelInfo import ModelInfo
from pathlib import Path

ROOT = Path('/petrobr/parceirosbr/spfm/joao.frare/logs+checkpoints_imagenet')
SAVE_FINETUNE_ROOT = Path('data/finetune')
SAVE_PRETRAIN_ROOT = Path('data/pretrain')

taxonomic_level_to_class = {
        9: 477,
        7: 200,
        6: 80,
        3: 10
}

taxonomic_x_default_metrics = Metrics([])
for level, num_classes in taxonomic_level_to_class.items():
        taxonomic_x_default_models = []
        taxonomic_x_default_models.append(ModelInfo(root_path= ROOT, reduction_mode= 'taxonomic', top_down= True, level= level, num_classes= num_classes))
        taxonomic_x_default_models.append(ModelInfo(root_path= ROOT, reduction_mode= 'default', num_classes= num_classes, per_class= 1300))
        metrics_same_class = Metrics(taxonomic_x_default_models)
        metrics_same_class.group_plot(
                title= f'Taxonomc ImageNet Pretrained models using {num_classes} classes Top-1 Acuraccy',
                save_path= SAVE_PRETRAIN_ROOT,
                y_axis= 'acc1',
                min_y= 0,
                max_y= 100,
                y_step= 5,
                x_axis= 'step',
                mul_factor= 100)
        
        taxonomic_x_default_metrics += metrics_same_class

taxonomic_x_default_metrics.individual_plot(
        title= f'Taxonomic ImageNet Pretrained models Train and Val Loss',
        save_path= SAVE_PRETRAIN_ROOT,
        y_axis= ['train_loss', 'val_loss'],
        x_axis= 'step',
        ncols= 4,
        yscale= 'log')
taxonomic_x_default_metrics.plot_taxonomic_x_default(
        title= f'Taxonomic - Pretrain Top 1 Acuraccy',
        save_path= SAVE_PRETRAIN_ROOT,
        pretrain= True
        )

#default configs
configs = {
    1000: [640, 320, 160, 80, 13],
    500:  [1300, 640, 320, 160],
    250:  [1300, 640, 320],
    125:  [1300, 640],
    10:   [1300],
}

#metricas de full dataset
full_dataset_model = ModelInfo(root_path= ROOT)
scratch_finetune_model = ModelInfo(root_path= ROOT, scratch= True)

metrics_default = {}
#preencher um métrics pra cada número de classes e plotar as informações de pré-treino
for num_classes in configs.keys():
        models = [full_dataset_model] if num_classes == 1000 else []
        
        for per_class in configs[num_classes]:
                models.append(ModelInfo(root_path= ROOT, reduction_mode= 'default', num_classes= num_classes, per_class= per_class))

        metrics_default[num_classes] = Metrics(models)
        metrics_default[num_classes].group_plot(
                title= f'ImageNet Pretrain, {num_classes} classes models Top-1 Acuraccy',
                save_path= SAVE_PRETRAIN_ROOT,
                y_axis= 'acc1',
                min_y= 0,
                max_y= 100,
                y_step= 5,
                x_axis= 'step',
                mul_factor= 100)

all_configurations = Metrics([])
for metric in metrics_default.values():
        all_configurations += metric

all_configurations.individual_plot(
        title= f'ImageNet Pretrained models, Train and Val Loss x Steps',
        save_path= SAVE_PRETRAIN_ROOT,
        y_axis= ['train_loss', 'val_loss'],
        x_axis= 'step',
        ncols= 4,
        yscale= 'log')
all_configurations.add_model(scratch_finetune_model)
all_configurations.plot_heatmap(
        title= 'Pretrain Models Top-1 Acuraccy',
        save_path= SAVE_PRETRAIN_ROOT,
        pretrain= True,
        cmap= 'Blues'
)

aliases = {
        'full_finetuning_deeplab': 'Full Finetuning',
        'full_freeze_linear': 'Linear Redout',
        'f3_N': 'F3',
        'seam_ai_N': 'Parihaka'
}

for dataset in full_dataset_model.datasets:
        for protocol in full_dataset_model.protocols:
                all_configurations.plot_heatmap(
                        title= f'Default mIoU, {aliases[protocol]} - {aliases[dataset]}',
                        save_path= SAVE_FINETUNE_ROOT,
                        dataset= dataset,
                        protocol= protocol)
                all_configurations.individual_plot(
                        title= f'Defaul Train and Validation loss curves, {aliases[protocol]} - {aliases[dataset]}',
                        save_path= SAVE_FINETUNE_ROOT,
                        y_axis= ['train_loss', 'val_loss'],
                        x_axis= 'epoch',
                        ncols= 4,
                        finetune= True,
                        dataset= dataset,
                        protocol= protocol
                )

                taxonomic_x_default_metrics.plot_taxonomic_x_default( 
                        title= f'Taxonomic mIoU, {protocol} - {dataset}',
                        save_path= SAVE_FINETUNE_ROOT,
                        dataset= dataset,
                        protocol= protocol)
                taxonomic_x_default_metrics.individual_plot(
                        title= f'Taxonomic Train and Validation loss curves, {aliases[protocol]} - {aliases[dataset]}',
                        save_path= SAVE_FINETUNE_ROOT,
                        y_axis= ['train_loss', 'val_loss'],
                        x_axis= 'epoch',
                        ncols= 4,
                        finetune= True,
                        dataset= dataset,
                        protocol= protocol
                )
        

