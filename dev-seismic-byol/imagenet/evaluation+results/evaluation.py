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
taxonomic_x_default_models = []
for level, num_classes in taxonomic_level_to_class.items():
        taxonomic_x_default_models.append(ModelInfo(root_path= ROOT, reduction_mode= 'taxonomic', top_down= True, level= level, num_classes= num_classes))
        taxonomic_x_default_models.append(ModelInfo(root_path= ROOT, reduction_mode= 'default', num_classes= num_classes, per_class= 1300))

taxonomic_x_default_metrics = Metrics(taxonomic_x_default_models)
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
                title= f'{num_classes} models Acc1 x Steps',
                save_path= SAVE_PRETRAIN_ROOT,
                y_axis= 'acc1',
                x_axis= 'step',
                mul_factor= 100)

all_configurations = Metrics([])
for metric in metrics_default.values():
        all_configurations += metric

all_configurations.individual_plot(
        title= f'models on Pretrain Train and Val Loss x Steps',
        save_path= SAVE_PRETRAIN_ROOT,
        y_axis= ['train_loss', 'val_loss'],
        x_axis= 'step',
        ncols= 4,
        yscale= 'log')
all_configurations.show_best_pretrain_metrics(filename= 'models best pretrain metrics', save_path= SAVE_PRETRAIN_ROOT)
all_configurations.add_model(scratch_finetune_model)

for dataset in full_dataset_model.datasets:
        for protocol in full_dataset_model.protocols:
                all_configurations.plot_heatmap(
                        title= f'models on {protocol} - {dataset}',
                        save_path= SAVE_FINETUNE_ROOT,
                        dataset= dataset,
                        protocol= protocol)
                taxonomic_x_default_metrics.plot_taxonomic_x_default( 
                        title= f'Default x Taxonomic - {protocol} - {dataset}',
                        save_path= SAVE_FINETUNE_ROOT,
                        dataset= dataset,
                        protocol= protocol)
        

