from metrics import Metrics
from modelInfo import ModelInfo
from pathlib import Path
ROOT = Path('/petrobr/parceirosbr/spfm/joao.frare/logs+checkpoints_imagenet')
SAVE_PRETRAIN_PATH = Path('data/pretrain')
SAVE_FINETUNE_PATH = Path('data/finetune')

def build_default_metrics(root_path: Path, datasets: list[str], protocols: list[str]):
        full_dataset = ModelInfo(
        root_path= root_path,
        reduction_mode= 'full', 
        datasets= datasets,
        protocols= protocols)

        default_metrics = Metrics([full_dataset])

        configs = {
        1000: [640, 320, 160, 80, 40, 20],
        500:  [1300, 640, 320, 160, 80, 40],
        250:  [1300, 640, 320, 160, 80],
        125:  [1300, 640, 320, 160],
        }

        for num_classes, per_class_list in configs.items():
                for per_class in per_class_list:
                        model_info = ModelInfo(
                                root_path= root_path,
                                reduction_mode= 'default',  
                                num_classes= num_classes, 
                                per_class= per_class,
                                datasets= datasets,
                                protocols= protocols)
                        default_metrics += model_info
        
        return default_metrics


def build_taxonomic_metrics(root_path: Path, datasets: list[str], protocols: list[str]):
        level_to_classes = {
                9: 477,
                7: 200,
                6: 80,
                3: 9
        }

        taxonomic_experiments = []

        for level, num_classes in level_to_classes.items():
                taxonomic = ModelInfo(
                        root_path = root_path,
                        reduction_mode = 'taxonomic',
                        top_down = True,
                        level= level,
                        num_classes= num_classes,
                        datasets= datasets,
                        protocols = protocols
                )
                default = ModelInfo(
                        root_path= root_path,
                        reduction_mode= 'default',
                        num_classes= num_classes,
                        per_class= 1300,
                        datasets= datasets,
                        protocols= protocols
                )
                taxonomic_experiments.extend([taxonomic, default])
        
        return Metrics(taxonomic_experiments)

def calculate_difference_tax_x_default(modern_tax_x_default, traditional_tax_x_default):
        difference_tax_x_default = {}
        for key in traditional_tax_x_default.keys():
                if key == 'default_std' or key == 'taxonomic_std':
                        difference_tax_x_default[key] = (modern_tax_x_default[key]**2 + traditional_tax_x_default[key]**2)**(1/2)
                elif key == 'default_mean' or key == 'taxonomic_mean':
                        difference_tax_x_default[key] = modern_tax_x_default[key] - traditional_tax_x_default[key]
                else:
                        difference_tax_x_default[key] = modern_tax_x_default[key]

        return difference_tax_x_default

def calculate_difference_heatmap(mean_modern, std_modern, mean_traditional, std_traditional):
        difference_mean = mean_modern - mean_traditional
        difference_std = (std_modern**2 + std_traditional**2)**(1/2)
        return (difference_mean, difference_std)

def build_infos(versions):
        infos = {}
        for version in versions:
                infos[version] = {
                        'mean_df': None,
                        'std_df': None,
                        'tax_x_default': None
                }
        return infos

datasets = ['seam_ai_N', 'f3_N']
protocols = ['full_finetuning_deeplab', 'full_freeze_linear']
versions = ['traditional', 'modern']

scratch_baseline = ModelInfo(ROOT, scratch= True, datasets= datasets, protocols= protocols)
default_metrics = build_default_metrics(root_path= ROOT, datasets= datasets, protocols= protocols)
taxonomic_metrics = build_taxonomic_metrics(root_path= ROOT, datasets= datasets, protocols= protocols)


#-----------------------------------------------------------------------pré-treino--------------------------------------------
pretrain_infos = build_infos(versions)
for version in versions:
        PRETRAIN_SAVE = SAVE_PRETRAIN_PATH / version 
        
        capitalized_version = version.capitalize()

        pretrain_infos[version]['mean_df'], pretrain_infos[version]['std_df'] = default_metrics.build_default_dataframe(version, True, None, None)
        pretrain_infos[version]['tax_x_default'] = taxonomic_metrics.get_tax_x_default(version, None, None, True)

        default_metrics.plot_heatmap(pretrain_infos[version]['mean_df'], pretrain_infos[version]['std_df'], f'{capitalized_version} Default Pretrain Top-1 Accuracy', PRETRAIN_SAVE, True, 'crest')
        taxonomic_metrics.plot_taxonomic_x_default(f'{capitalized_version} Taxonomic Models Pretrain Top-1 Accuracy', PRETRAIN_SAVE, pretrain_infos[version]['tax_x_default'], True, False)

difference_heatmap_mean, difference_heatmap_std = calculate_difference_heatmap(pretrain_infos['modern']['mean_df'], pretrain_infos['modern']['std_df'], pretrain_infos['traditional']['mean_df'], pretrain_infos['traditional']['std_df'])
difference_tax_x_default = calculate_difference_tax_x_default(pretrain_infos['modern']['tax_x_default'], pretrain_infos['traditional']['tax_x_default'])

default_metrics.plot_heatmap(difference_heatmap_mean, difference_heatmap_std, 'Modern - Traditional default', SAVE_PRETRAIN_PATH / 'difference', True, 'viridis', True)
taxonomic_metrics.plot_taxonomic_x_default('Modern - Traditional taxonomic', SAVE_PRETRAIN_PATH / 'difference', difference_tax_x_default, True, True)

df = taxonomic_metrics.get_miou_df(pretrain= True)
df['difference'] = (
    (df['modern mean acc1'] - df['traditional mean acc1']) * 100
).map('{:.2f}'.format)

df['modern acc1'] = (
    (df['modern mean acc1'] * 100).map('{:.2f}'.format)
    + " ± "
    + (df['modern std acc1'] * 100).map('{:.2f}'.format)
)

df['traditional acc1'] = (
    (df['traditional mean acc1'] * 100).map('{:.2f}'.format)
    + " ± "
    + (df['traditional std acc1'] * 100).map('{:.2f}'.format)
)

df = df.drop(columns= ['modern mean acc1', 'traditional mean acc1', 'modern std acc1', 'modern std acc1'], axis= 1)
df.to_csv(SAVE_PRETRAIN_PATH / 'tax_models.txt')



#------------------------------------------------------------------finetune-------------------------------------------------------
default_metrics += scratch_baseline
taxonomic_metrics += scratch_baseline
for dataset in datasets:
        for protocol in protocols:
                FINETUNE_SAVE = SAVE_FINETUNE_PATH / dataset / protocol
                traditional_heatmap_mean, traditional_heatmap_std = default_metrics.build_default_dataframe(version= 'traditional', pretrain= False, dataset= dataset, protocol= protocol)
                default_metrics.plot_heatmap(traditional_heatmap_mean, traditional_heatmap_std,f'{dataset} {protocol} traditional default', FINETUNE_SAVE / 'traditional', False)

                modern_heatmap_mean, modern_heatmap_std = default_metrics.build_default_dataframe(version= 'modern', pretrain= False, dataset= dataset, protocol= protocol)
                default_metrics.plot_heatmap(modern_heatmap_mean, modern_heatmap_std,f'{dataset} {protocol} modern default', FINETUNE_SAVE / 'modern', False)

                traditional_tax_x_default = taxonomic_metrics.get_tax_x_default(version= 'traditional', dataset= dataset, protocol= protocol, pretrain= False)
                taxonomic_metrics.plot_taxonomic_x_default(f'{dataset} {protocol} traditional taxonomic', FINETUNE_SAVE / 'traditional', traditional_tax_x_default, False, False)

                modern_tax_x_default = taxonomic_metrics.get_tax_x_default(version= 'modern', dataset= dataset, protocol= protocol, pretrain= False)
                taxonomic_metrics.plot_taxonomic_x_default(f'{dataset} {protocol} modern taxonomic', FINETUNE_SAVE / 'modern', modern_tax_x_default, False, False)

                difference_heatmap_mean = modern_heatmap_mean - traditional_heatmap_mean
                difference_heatmap_std = (modern_heatmap_std**2 + traditional_heatmap_std**2)**(1/2)
                difference_tax_x_default = calculate_difference_tax_x_default(modern_tax_x_default, traditional_tax_x_default)

                default_metrics.plot_heatmap(difference_heatmap_mean, difference_heatmap_std, 'Modern - Traditional default config', FINETUNE_SAVE / 'difference', False, 'magma', True)
                default_metrics.plot_taxonomic_x_default('Modern - Traditional tax config', FINETUNE_SAVE / 'difference', difference_tax_x_default, False, True)

                df = taxonomic_metrics.get_miou_df(dataset, protocol, False)
                df['difference'] = (
                (df['modern mean mIoU'] - df['traditional mean mIoU']) * 100
                ).map('{:.2f}'.format)

                df['modern mIoU'] = (
                (df['modern mean mIoU'] * 100).map('{:.2f}'.format)
                + " ± "
                + (df['modern std mIoU'] * 100).map('{:.2f}'.format)
                )

                df['traditional mIoU'] = (
                (df['traditional mean mIoU'] * 100).map('{:.2f}'.format)
                + " ± "
                + (df['traditional std mIoU'] * 100).map('{:.2f}'.format)
                )

                df = df.drop(columns= ['modern mean mIoU', 'traditional mean mIoU', 'modern std mIoU', 'traditional std mIoU'], axis= 1)
                df.to_csv(FINETUNE_SAVE / 'tax_models.txt')
                


                

        

