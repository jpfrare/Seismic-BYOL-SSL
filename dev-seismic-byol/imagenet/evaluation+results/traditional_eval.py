from metrics import Metrics
from modelInfo import ModelInfo
from pathlib import Path
ROOT = Path('/petrobr/parceirosbr/spfm/joao.frare/logs+checkpoints_imagenet')
SAVE_PATH = Path('data/traditional/pretrain')

full_dataset = ModelInfo(
    root_path= ROOT,
    reduction_mode= 'full', 
    top_down= False, 
    level= 0,
    version= 'traditional',  
    num_classes= 1000, 
    per_class= 1300, 
    scratch= False, 
    torchPretrained= False,
    datasets= [],
    protocols= [])

default_trad_metrics = Metrics([full_dataset])

configs = {
    1000: [640, 320, 160, 80, 40, 20],
    500:  [1300, 640, 320, 160, 80, 40],
    250:  [1300, 640, 320, 160, 80],
    125:  [1300, 640, 320, 160],
}

for num_classes, per_class_list in configs.items():
        for per_class in per_class_list:
                model_info = ModelInfo(
                        root_path= ROOT,
                        reduction_mode= 'default',  
                        num_classes= num_classes, 
                        per_class= per_class, 
                        datasets=[],
                        protocols=[])
                default_trad_metrics += model_info



default_trad_metrics.individual_plot(
        title= 'Traditional Models Train and Val Loss x Steps',
        save_path= SAVE_PATH,
        y_axis= ['train_loss', 'val_loss'],
        x_axis= 'step',
        ncols= 4,
        yscale= 'log')

default_trad_metrics.plot_heatmap(
        title= 'Traditional Pretrain Top-1 Accuracy',
        save_path= SAVE_PATH,
        pretrain= True,
        cmap= 'crest'
)

