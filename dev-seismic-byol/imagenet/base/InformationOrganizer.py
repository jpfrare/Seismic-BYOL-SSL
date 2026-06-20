import argparse
import os
from pathlib import Path
from .ImagenetReader import ImagenetReader, ImagenetValReader

    
class TrainOrganizer():
    task: str
    data_root: str
    ckpt_dir: Path
    log_dir: Path
    model_name: str
    parser: argparse.ArgumentParser
    args: argparse.Namespace

    def __init__(self, data_root: str): #data_root -> onde está a pasta de logs+checkpoints
        self.data_root = data_root
        self.task = 'Train'
        self._parse_args()
        dirs = self._get_dirs_and_set_model_name()

        self.ckpt_dir = dirs/'checkpoints'
        self.log_dir = dirs/'logs'

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    
    def _set_up_parser(self):
        self.parser = argparse.ArgumentParser(description="parser for Imagenet Train")

        # Primeiro argumento: como faríamos a redução do dataset
        self.parser.add_argument(
            "--reduction_mode", 
            type=str, 
            choices=['taxonomic', 'default', 'full'], 
            required=True, 
            help='como diminuiremos a diversidade'
        )
        self.parser.add_argument("--repetition", type= int, default= 42, help= 'número da repetição')

        # Modo taxonomico
        self.parser.add_argument("--top_down", action="store_true", help='padrão: fará um bottom_up')
        self.parser.add_argument("--level", type=int, default=3, help='nível de corte')

        # Modo default
        self.parser.add_argument("--num_classes", type=int, default=1000, help='número de classes utilizados')
        self.parser.add_argument("--per_class", type=int, default=1300, help='número de imagens por classe utilizados')

        self.parser.add_argument("--test", action= "store_true", help= 'teste final')
    
    def _get_dirs_and_set_model_name(self):
        dirs = Path(self.data_root)/self.task/f'{self.args.repetition}'/self.args.reduction_mode
        mode = self.args.reduction_mode

        if mode == 'taxonomic':
            strategy = 'top_down' if self.args.top_down else 'bottom_up'
            self.model_name = f'{strategy}_level_{self.args.level}'
            dirs = dirs/self.model_name
            
        elif mode == 'default':
            self.model_name = f'num_classes_{self.args.num_classes}_per_class_{self.args.per_class}'
            dirs = dirs/self.model_name
        else:
            self.model_name = 'full'
        
        return dirs
    
    def __str__(self):
        info = ['-------Train Infos:---------',
        f'task: {self.task}',
        f'ckpt_dir: {self.ckpt_dir}',
        f'log_dir: {self.log_dir}',
        f'model_name: {self.model_name}',
        '-------ArgParser Arguments---']
        for key, value in vars(self.args).items():
            info.append(f'{key}: {value}')
        
        return '\n'.join(info)

    def _parse_args(self):
        self._set_up_parser()
        self.args = self.parser.parse_args()

        # Inconsistência 1: Se o usuário está no modo 'default', ele NÃO deve mexer em flags taxonômicas
        if self.args.reduction_mode == 'default':
            # Se ele passou --top_down (que vira True) ou mudou o --level padrão
            if self.args.top_down or self.args.level != 3:
                self.parser.error("Inconsistência! No modo 'default', os argumentos --top_down e --level não têm efeito e não devem ser alterados.")

        # Inconsistência 2: Se ele está no modo 'taxonomic', não faz sentido alterar limites numéricos/aleatórios
        elif self.args.reduction_mode == 'taxonomic':
            if self.args.num_classes != 1000 or self.args.per_class != 1300:
                self.parser.error(
                    "Inconsistência! No modo 'taxonomic', a redução é feita via árvore do WordNet. Os argumentos --num_classes e --per_class não devem ser alterados.")
            if self.args.level < 0 or self.args.level > 10:
                self.parser.error(f"Erro! O --level fornecido ({self.args.level}) é inválido para a estrutura do WordNet. Escolha um valor entre 0 e 10.")

        elif self.args.reduction_mode == 'full':
            print('Using ImageNet full dataset!')
    
    def set_readers(self, train_dataset_root, train_entries, val_root, ground_truth_root, mat_root):
        train_reader = ImagenetReader(train_dataset_root, train_entries)
        val_reader = ImagenetValReader(val_root, ground_truth_root, mat_root)

        if self.args.reduction_mode == 'taxonomic':
            #fará o agrupamento top down ou bottom up
            mode = 'Top Down' if self.args.top_down else 'Bottom Up'
            self.args.num_classes = val_reader.to_coarse_classes(top_down= self.args.top_down, level= self.args.level, mat_path= mat_root)
            train_reader.to_coarse_classes(top_down= self.args.top_down, level= self.args.level, mat_path= mat_root)

            print(f'Using {self.args.num_classes} after {mode} clustering!')
        
        return (train_reader, val_reader)

class FinetuningOrganizer(TrainOrganizer):
    finetune_model_name: str
    finetune_log_dir: Path
    finetune_ckpt_dir: Path

    def __init__(self, data_root: str):
        super().__init__(data_root)
        self.task = 'Finetune'

        dirs= self._get_finetune_dirs_and_set_model_name()

        self.finetune_ckpt_dir = dirs/'checkpoints'
        self.finetune_log_dir = dirs/'logs'

        self.finetune_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.finetune_log_dir.mkdir(parents=True, exist_ok=True)

    def _set_up_parser(self):
        super()._set_up_parser()
        self.parser.add_argument("--scratch", action= 'store_true', help= 'Train From Scratch')
        self.parser.add_argument(
            "--backbone_freeze",
            type= str,
            choices= ['full_freeze', 'custom_freeze', 'full_finetuning'],
            required= True,
            help= 'como se dará os pesos do backbone'
        )
        self.parser.add_argument(
            "--pred_head",
            type= str,
            choices= ['linear', 'deeplab'],
            required= True,
            help= 'qual cabeça será usada'
        )
        self.parser.add_argument("--finetune_dataset", type= str, choices= ['f3_N', 'seam_ai_N'], required= True, help= 'dataset de finetune')

    def _get_finetune_dirs_and_set_model_name(self):
        if self.args.scratch:
            self.model_name = 'scratch'
            dirs = Path(self.data_root)/self.task/f'{self.args.repetition}'/'scratch'
        else:
            dirs =  super()._get_dirs_and_set_model_name()
        dirs = dirs / f'finetune_{self.args.finetune_dataset}'/ f'{self.args.backbone_freeze}_{self.args.pred_head}'
        self.finetune_model_name = f'{self.model_name}_finetune_{self.args.finetune_dataset}'
        return dirs
    
    def __str__(self):
        sup_info = super().__str__()

        info = ['-------Finetune Infos:---------',
        f'finetune_log_dir: {self.finetune_log_dir}',
        f'finetune_ckpt_dir: {self.finetune_ckpt_dir}',
        f'finetune_model_name: {self.finetune_model_name}']
        
        info = '\n'.join(info)

        return sup_info + '\n' +  info
    