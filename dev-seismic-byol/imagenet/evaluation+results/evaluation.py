from metrics import Metrics
from modelInfo import ModelInfo
from pathlib import Path
ROOT = Path('/petrobr/parceirosbr/spfm/joao.frare/logs+checkpoints_imagenet')
SAVE_PRETRAIN_PATH = Path('data/pretrain')
SAVE_FINETUNE_PATH = Path('data/finetune')

def build_default_metrics(root_path: Path, version: str, datasets: list[str], protocols: list[str]):
        full_dataset = ModelInfo(
        root_path= root_path,
        reduction_mode= 'full', 
        version= version,  
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
                                version= version, 
                                datasets= datasets,
                                protocols= protocols)
                        default_metrics += model_info
        
        return default_metrics


def build_taxonomic_metrics(root_path: Path, version: str, datasets: list[str], protocols: list[str]):
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
                        version= version,
                        top_down = True,
                        level= level,
                        num_classes= num_classes,
                        datasets= datasets,
                        protocols = protocols
                )
                default = ModelInfo(
                        root_path= root_path,
                        reduction_mode= 'default',
                        version= version,
                        num_classes= num_classes,
                        per_class= 1300,
                        datasets= datasets,
                        protocols= protocols
                )
                taxonomic_experiments.extend([taxonomic, default])
        
        return Metrics(taxonomic_experiments)

datasets = ['seam_ai_N', 'f3_N']
protocols = ['full_finetuning_deeplab', 'full_freeze_linear']

scratch_baseline = ModelInfo(ROOT, scratch= True, datasets= datasets, protocols= protocols)


for version in ['traditional', 'modern']:

        #pre-treino
        PRETRAIN_SAVE = SAVE_PRETRAIN_PATH / version 
        
        capitalized_version = version.capitalize()

        default_metrics = build_default_metrics(root_path= ROOT, version= version, datasets= datasets, protocols= protocols)

        default_metrics.individual_plot(
                title= f'{capitalized_version} Default Models Train and Val Loss x Steps',
                save_path= PRETRAIN_SAVE,
                y1_axis= ['train_loss', 'val_loss'],
                y2_axis= ['acc1'],
                y2_limits= (0, 100),
                mul_factor2= 100,
                x_axis= 'step',
                ncols= 4,
                y1scale= 'log',
        )

        default_metrics.plot_heatmap(
                title= f'{capitalized_version} Default Pretrain Top-1 Accuracy',
                save_path= PRETRAIN_SAVE,
                pretrain= True,
                pretrain_key= 'Mean',
                cmap= 'crest'
        )

        taxonomic_metrics = build_taxonomic_metrics(root_path= ROOT, version= version, datasets= datasets, protocols= protocols)

        taxonomic_metrics.individual_plot(
                title= f'{capitalized_version} Taxonomic Models Train and Val Loss x Steps',
                save_path= PRETRAIN_SAVE,
                y1_axis= ['train_loss', 'val_loss'],
                y2_axis= ['acc1'],
                y2_limits= (0, 100),
                mul_factor2= 100,
                x_axis= 'step',
                ncols= 2,
                y1scale= 'log',
        )

        taxonomic_metrics.plot_taxonomic_x_default(
                title= f'{capitalized_version} Taxonomic Models Pretrain Top-1 Accuracy',
                save_path= PRETRAIN_SAVE,
                pretrain = True,
                pretrain_key= 'Mean'
        )

        default_metrics += scratch_baseline
        taxonomic_metrics += scratch_baseline

        #finetuning
        for dataset in datasets:
                for protocol in protocols:
                        FINETUNE_SAVE = SAVE_FINETUNE_PATH / dataset / protocol
                        

        

