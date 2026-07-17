from pathlib import Path
import os
import pandas as pd
import numpy as np
import copy
import yaml

class ModelInfo():
    root_path: Path                                                         #-> pasta raiz dos dados
    pretrain_path: Path                                                     #-> caminho a partir da pasta raiz e repetição que leva aos dados do modelo
    finetune_path: dict[str : dict[str : Path]]                             #-> caminhos a partir da pasta raiz e repetição que levam as combinações de 
                                                                            #datasets e protocolos de finetuning

    pretrain_dataframe: pd.DataFrame                                        #-> dataframe que contém os dados agregados das reeptições do pré-treino por step
    finetune_dataframes: dict[str, dict[str, pd.DataFrame]]                 #-> dataframes que contém todos os dados agregados das repetiões de todas as combinações de datasets e protocolos do finetuning por época
    finetune_miou: dict[str, dict[str, dict[str, float]]]                   #-> valores (média e desvio padrão) de todas as combinações de datasets e protocolos do finetuning

    model_name: str                                                         #-> nome do modelo a ser exibido
    pretrained_classes: int                                                 #-> número de classes usadas no pré-treino
    pretrained_images: int                                                  #-> número de imagens usadas no pré-treino
    reduction_mode: str                                                     #-> como de deu o modelo de treino

    datasets: list[str]                                                     #-> datasets usados no modelo
    protocols: list[str]                                                    #-> protocolos utlizados

    def __init__(self, 
    root_path: Path,
    reduction_mode: str = 'full', 
    top_down: bool = False, 
    level: int = 0, 
    num_classes: int = 1000, 
    per_class: int = 1300, 
    scratch: bool = False, 
    torchPretrained: bool = False,
    datasets: list = ['seam_ai_N', 'f3_N'],
    protocols: list = ['full_finetuning_deeplab', 'full_freeze_linear']):

        '''inicia o objeto, inserindo os miolos de caminho corretamente nas variáveis de caminho, atribuindo número de classes e imagens de pré-treino e o
        nome do modelo'''

        self.finetune_path = self._create_dataset_protocols_dictionary(datasets, protocols)
        self.datasets = datasets
        self.protocols = protocols

        self.root_path = root_path
        self.torchPretrained = torchPretrained
        self.scratch = scratch

        if scratch:
            self.pretrained_images = 0
            self.pretrained_classes = 0
            self.model_name = 'Scratch'

            self.pretrain_path = None
            base = Path('scratch')
            self._set_finetune_paths(base)

        elif torchPretrained:
            self.pretrained_images = 1_280_000
            self.pretrained_classes = 1000
            self.model_name = 'Torch Pretrained'

            self.pretrain_path = None
            self.finetune_path = None

        elif reduction_mode == 'full':
            self.pretrained_images = 1_280_000
            self.pretrained_classes = 1000
            self.model_name = 'Full Dataset'

            base = Path('full') 
            self.pretrain_path = base / 'logs'
            self._set_finetune_paths(base)
            
        elif reduction_mode == 'default':
            real_per_class = min(per_class, 1280)
            self.pretrained_images = real_per_class * num_classes
            self.pretrained_classes = num_classes

            if num_classes == 10 and per_class == 1300:
                self.pretrained_images = 13000

            self.model_name = f'{num_classes}C, {real_per_class}IpC'

            base = Path('default') / f'num_classes_{num_classes}_per_class_{per_class}'
            self.pretrain_path = base / 'logs'
            self._set_finetune_paths(base)
        
        elif reduction_mode == 'taxonomic':
            self.pretrained_images = 1_280_000
            self.pretrained_classes = num_classes
            cut_mode = 'top_down' if top_down else 'bottom_up'
            self.model_name = f'Level_{level}_(C_{num_classes})'

            base = Path('taxonomic') / f'{cut_mode}_level_{level}'
            self.pretrain_path = base / 'logs'
            self._set_finetune_paths(base)

        else:
            raise ValueError('Valores inseridos incongruentes!')
        
        self._load_data()

    def _create_dataset_protocols_dictionary(self, datasets: list[str], protocols: list[str], start_value = None) -> dict:
        '''cria um dicionário aninhado'''
        ans = {}
    
        for dataset in datasets:
            ans[dataset] = {}
            for protocol in protocols:
                ans[dataset][protocol] = copy.deepcopy(start_value) if start_value is not None else start_value
        
        return ans
    
    def _set_finetune_paths(self, base: Path) -> None:
        for dataset in self.datasets:
            for protocol in self.protocols:
                self.finetune_path[dataset][protocol] = (
                    base / f"finetune_{dataset}" / protocol / "logs"
                )
    
    def _read_csv(self, csv_path: Path, dropna_subset: list[str]) -> pd.DataFrame:
        data_frame = pd.read_csv(csv_path)
        data_frame = data_frame.groupby("epoch", as_index= False).agg({
            value: 'max' for value in dropna_subset
        })
        data_frame = data_frame.dropna(subset = dropna_subset)
        return data_frame
        

    def _load_data(self) -> None:
        '''carrega todos os dados para as respectivas variáveis'''

        self.finetune_dataframes = self._create_dataset_protocols_dictionary(self.datasets, self.protocols, start_value= [])
        self.finetune_miou = self._create_dataset_protocols_dictionary(self.datasets, self.protocols, start_value= [])
        self.pretrain_dataframe = []

        if self.torchPretrained:
            #no caso de ser um modelo pré-treinado no torch, ja temos os valores dos resultados do vinícius para esse conjunto de datasets e protocolos
            self.finetune_miou['f3_N']['full_finetuning_deeplab'] = {
                'mean': 0.75,
                'std': 0.01,
            }
            self.finetune_miou['f3_N']['full_freeze_linear'] = {
                'mean': 0.47,
                'std': 0.00,
            }
            self.finetune_miou['seam_ai_N']['full_finetuning_deeplab'] = {
                'mean': 0.72,
                'std': 0.01,
            }
            self.finetune_miou['seam_ai_N']['full_freeze_linear'] = {
                'mean': 0.36,
                'std': 0.00,
            }

            self.pretrain_dataframe = pd.DataFrame()
            for dataset in self.datasets:
                for protocol in self.protocols:
                    self.finetune_dataframes[dataset][protocol] = pd.DataFrame()
            return

        for repetition in range(3):

            if not self.scratch:
                #--------------------------lendo repetições do pré-treino----------------------------------------------------------
                pretrain_csv_path = self.root_path / 'Train' / f'{repetition}' / self.pretrain_path / 'metrics.csv'
                self.pretrain_dataframe.append(self._read_csv(pretrain_csv_path, ['step', 'train_loss_epoch', 'val_acc1', 'val_acc5', 'val_loss']))

            #--------------------------------lendo repetições do finetuning---------------------------------------------------------
            for dataset in self.datasets:
                for protocol in self.protocols:

                    finetune_path = self.root_path / 'Finetune' / f'{repetition}' / self.finetune_path[dataset][protocol]
                    self.finetune_dataframes[dataset][protocol].append(self._read_csv(finetune_path / 'metrics.csv', ['train_loss', 'val_loss']))

                    finetune_yaml_path = next(finetune_path.glob('metrics*.yaml')) 
                    with open(finetune_yaml_path, 'r') as file:
                        data = yaml.safe_load(file)
                        miou = data['classification']['mIoU'][0]
                        self.finetune_miou[dataset][protocol].append(miou)
        
        if not self.scratch:
            #----------------------------------------agregando repetições do pré-treino----------------------------------------------
            self.pretrain_dataframe = pd.concat(self.pretrain_dataframe)
            self.pretrain_dataframe = self.pretrain_dataframe.groupby('step').agg(
                    mean_val_loss= ('val_loss', 'mean'),
                    std_val_loss= ('val_loss', 'std'),
                    mean_train_loss = ('train_loss_epoch', 'mean'),
                    std_train_loss = ('train_loss_epoch', 'std'),
                    mean_acc1= ('val_acc1', 'mean'),
                    std_acc1= ('val_acc1', 'std'),
                    mean_acc5= ('val_acc5', 'mean'),
                    std_acc5= ('val_acc5', 'std') 
                ).reset_index()
        
        else:
            self.pretrain_dataframe = pd.DataFrame()
        
        #-----------------------------------------agregando repetições do finetuning------------------------------------------------
        for dataset in self.datasets:
            for protocol in self.protocols:
                self.finetune_dataframes[dataset][protocol] = pd.concat(self.finetune_dataframes[dataset][protocol])
                self.finetune_dataframes[dataset][protocol] = self.finetune_dataframes[dataset][protocol].groupby('epoch').agg(
                    mean_val_loss= ('val_loss', 'mean'),
                    std_val_loss= ('val_loss', 'std'),
                    mean_train_loss = ('train_loss', 'mean'),
                    std_train_loss = ('train_loss', 'std')
                ).reset_index()

                mean = np.mean(self.finetune_miou[dataset][protocol])
                std = np.std(self.finetune_miou[dataset][protocol], ddof= 1)

                self.finetune_miou[dataset][protocol] = {
                    'mean': mean,
                    'std': std,
                }
        return
    
    def get_pretrain_dataframe(self) -> pd.DataFrame:
        return self.pretrain_dataframe
    
    def get_finetune_dataframe(self, dataset: str, protocol: str) -> pd.DataFrame:
        return self.finetune_dataframes[dataset][protocol]

    def get_finetune_miou(self, dataset: str, protocol: str) -> tuple[float, float]:
        '''return (mean, std) for desired dataset + protocol'''
        data = self.finetune_miou[dataset][protocol]
        return (data['mean'], data['std'])