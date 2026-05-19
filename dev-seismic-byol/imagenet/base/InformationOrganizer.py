import argparse
import os
from pathlib import Path

class Organizer():
    task: str
    data_root: str
    ckpt_dir: Path
    log_dir: Path
    model_name: str
    args: argparse.Namespace

    def __init__(self, task: str, data_root: str):
        self.task = task
        self.data_root = data_root
    
    def _parse_args(self):
        raise NotImplementedError
    
    def get_model_name(self):
        raise NotImplementedError

class TrainOrganizer(Organizer):

    def __init__(self, data_root: str):
        super().__init__('Train', data_root)
        self._parse_args()

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
        
        self.ckpt_dir = dirs/'checkpoints'
        self.log_dir = dirs/'logs'

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    
    def _parse_args(self):
        parser = argparse.ArgumentParser(description="parser for Imagenet Train")

        # Primeiro argumento: como faríamos a redução do dataset
        parser.add_argument(
            "--reduction_mode", 
            type=str, 
            choices=['taxonomic', 'default', 'full'], 
            required=True, 
            help='como diminuiremos a diversidade'
        )
        parser.add_argument("--repetition", type= int, default= 42, help= 'número da repetição')

        # Modo taxonomico
        parser.add_argument("--top_down", action="store_true", help='padrão: fará um bottom_up')
        parser.add_argument("--level", type=int, default=3, help='nível de corte')

        # Modo default
        parser.add_argument("--num_classes", type=int, default=1000, help='número de classes utilizados')
        parser.add_argument("--per_class", type=int, default=1300, help='número de imagens por classe utilizados')

        args = parser.parse_args()


        # Inconsistência 1: Se o usuário está no modo 'default', ele NÃO deve mexer em flags taxonômicas
        if args.reduction_mode == 'default':
            # Se ele passou --top_down (que vira True) ou mudou o --level padrão
            if args.top_down or args.level != 3:
                parser.error("Inconsistência! No modo 'default', os argumentos --top_down e --level não têm efeito e não devem ser alterados.")

        # Inconsistência 2: Se ele está no modo 'taxonomic', não faz sentido alterar limites numéricos/aleatórios
        elif args.reduction_mode == 'taxonomic':
            if args.num_classes != 1000 or args.per_class != 1300:
                parser.error(
                    "Inconsistência! No modo 'taxonomic', a redução é feita via árvore do WordNet. Os argumentos --num_classes e --per_class não devem ser alterados.")
            if args.level < 0 or args.level > 10:
                parser.error(f"Erro! O --level fornecido ({args.level}) é inválido para a estrutura do WordNet. Escolha um valor entre 0 e 10.")

        elif args.reduction_mode == 'full':
            print('Using ImageNet full dataset!')

        self.args = args
        