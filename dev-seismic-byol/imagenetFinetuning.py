import argparse
from pathlib import Path

from functions import *

parser = argparser.ArgumentParser(
    description= "Finetuning on Imagenet"
)

parser.add_argument(
    "per_class", type= int, description= "per_class_model used in pretrain"
)

parser.add_argument(
    "repetition", type= int, description= "repetition used"
)

parser.add_argument(
    "finetning_dataset", type= str, description= "dataset used in finetuning (seam_ai_N or f3_N)"
)

args = parser.parse_args()

#------------------------------------PATHS---------------------------
root = '/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/'

model_name = f"V{args.repetition}_pretrain_imagenet_{args.per_class}_per_class"
mapping = get_dataset_mapping()
pretrain_ckpt_path = f"{root}/checkpoints/ckpt_vinicius/pretrain/{args.repetition}/{model_name}/imagenet"
log_path

#----------------------------------MODELO---------------------------
deeplab_backbone = 