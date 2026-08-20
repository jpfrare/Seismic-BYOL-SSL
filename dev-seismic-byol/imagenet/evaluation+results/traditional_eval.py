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

metrics = Metrics([full_dataset])

metrics.individual_plot(
        title= 'models trans x val_loss',
        save_path= SAVE_PATH,
        y_axis= ['train_loss', 'val_loss'],
        x_axis= 'step',
        ncols= 1,
        yscale= 'log')

metrics.group_plot(
        title= f'ImageNet Pretrain, 1000 classes models Top-1 Acuraccy',
        save_path= SAVE_PATH,
        y_axis= 'acc1',
        y_axis_label= 'Top-1 Acuraccy (%)',
        min_y= 0,
        max_y= 100,
        y_step= 5,
        x_axis= 'step',
        x_axis_label= 'Step',
        mul_factor= 100)