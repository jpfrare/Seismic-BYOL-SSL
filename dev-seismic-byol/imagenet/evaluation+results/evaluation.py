from metrics import Metrics
from modelInfo import ModelInfo
from pathlib import Path
ROOT = Path('/petrobr/parceirosbr/spfm/joao.frare/logs+checkpoints_imagenet')
SAVE_PATH = Path('data/pretrain')

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



modern_full = ModelInfo(
        root_path= ROOT,
        reduction_mode= 'full', 
        version= 'modern',  
        datasets= [],
        protocols= [])

traditional_full = ModelInfo(
        root_path= ROOT,
        reduction_mode= 'full', 
        version= 'traditional',  
        datasets= [],
        protocols= [])

modern_few =  ModelInfo(
        root_path= ROOT,
        reduction_mode= 'default',  
        num_classes= 125, 
        per_class= 1300,
        version= 'modern', 
        datasets= [],
        protocols= [])

tradidional_few = ModelInfo(
        root_path= ROOT,
        reduction_mode= 'default',  
        num_classes= 125, 
        per_class= 1300,
        version= 'traditional', 
        datasets= [],
        protocols= [])

metrics = Metrics([])

metrics.compare_acc_histogram(traditional_model= traditional_full, modern_model= modern_full, save_path= Path('./data'), name= 'Full Dataset Hstogram')
metrics.compare_acc_histogram(traditional_model= tradidional_few, modern_model= modern_few, save_path= Path('./data'), name= 'Few Dataset Hstogram')
metrics.compare_acc_per_class(modern_model= modern_full, traditional_model= traditional_full, save_path= Path('./data'), name= 'Full Dataset Per Class')
metrics.compare_acc_per_class(modern_model= modern_few, traditional_model= tradidional_few, save_path= Path('./data'), name= '125C 1300 IpC Per Class')

#pretrain
for version in ['traditional', 'modern']:
        PRETRAIN_SAVE = SAVE_PATH / version 
        capitalized_version = version.capitalize()

        default_metrics = build_default_metrics(root_path= ROOT, version= version, datasets= [], protocols= [])

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

        taxonomic_metrics = build_taxonomic_metrics(root_path= ROOT, version= version, datasets= [], protocols= [])

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

        

