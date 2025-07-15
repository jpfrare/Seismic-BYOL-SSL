from minerva.models.ssl.byol import BYOL
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone
from minerva.models.loaders import FromPretrained
from torchvision.models.resnet import resnet50
from minerva.models.nets.image.deeplabv3 import DeepLabV3

from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import torchvision.models
import torch
import torchmetrics
import torch.nn as nn

import argparse
from evaluate_copy import main  # seu main de avaliação
from functions import *
from pathlib import Path
import re
import pandas as pd


def extract_step_number(filename):
    match = re.search(r"step=(\d+)", filename)
    return int(match.group(1)) if match else None

# def extract_epoch_number(filename):
#     match = re.search(r"epoch=(\d+)", filename)
#     return int(match.group(1)) if match else None

def get_models_files(base_dir="./ht_ckpt/train", target_repetition=None):
    base_dir = Path(base_dir)
    results = []

    repetitions = (
        [str(target_repetition)]
        if target_repetition is not None
        else [d.name for d in base_dir.iterdir() if d.is_dir()]
    )

    for repetition_dir in repetitions:
        rep_path = base_dir / repetition_dir
        if not rep_path.is_dir():
            continue

        for model_dir in rep_path.iterdir():
            if not model_dir.is_dir():
                continue

            match = re.match(
                r"finetune_V(\d+)_pretrain_(.+?)_In(\d+)_B(\d+)_E(\d+)_lr([\deE\.-]+)_step_(\d+)",
                model_dir.name,
            )
            # match = re.match(
            #     r"finetune_V(\d+)_pretrain_(.+?)_In(\d+)_B(\d+)_E(\d+)_lr([\deE\.-]+)_epoch_(\d+)",
            #     model_dir.name,
            # )
            if not match:
                continue

            _, pretrain_data, input_size, batch_size, epochs, learning_rate, ckpt_epoch = match.groups()

            for inner_dir in model_dir.iterdir():
                if not inner_dir.is_dir():
                    continue

                # Agora procuramos arquivos tipo epoch=13-step=1960.ckpt
                # ckpt_files = [
                #     f for f in inner_dir.iterdir()
                #     if f.is_file() and re.match(r"epoch=\d+(-step=\d+)?\.ckpt", f.name)
                # ]
                ckpt_files = [
                    f for f in inner_dir.iterdir()
                    if f.is_file() and re.match(r"epoch=\d+(-step=\d+)?\.ckpt", f.name)
                ]

                for ckpt_file in ckpt_files:
                    epoch_save = extract_epoch_number(ckpt_file.name)
                    results.append(
                        {
                            "model_name": model_dir.name,
                            "combination": int(repetition_dir),
                            "pretrain_data": pretrain_data,
                            "input_size": int(input_size),
                            "batch_size": int(batch_size),
                            "epochs": int(epochs),
                            "learning_rate": float(learning_rate),
                            "ckpt_file": str(ckpt_file),
                            "epoch_save": epoch_save,
                            "epoch_trained": int(ckpt_epoch),
                        }
                    )

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for seismic segmentation models."
    )
    parser.add_argument(
        "--combination",
        type=int,
        required=True,
        help="Experiment combination index to evaluate."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="List of GPU indices to use."
    )
    
    ckpt_path = "/home/vinicius.soares/Seismic-Byol/dev-seismic-byol/ht_ckpt/train_02_unfreeze"

    args = parser.parse_args()

    TEST_LOGS_PATH = f"ht_logs/test_02_unfreeze_rerun/{args.combination}"
    TEST_CKPT_PATH = f"ht_ckpt/test_02_unfreeze_rerun/{args.combination}"
    
    logger.info(f"Target combination: {args.combination}")
    
    models_list = get_models_files(
        base_dir = ckpt_path,
        target_repetition=args.combination,
        )
    
    df = pd.DataFrame(models_list)
    print(df.head)
    
    logger.info(f"Ammount of models found: {len(models_list)}")
    
    # Filter models based on user input
    parser.add_argument(
        "--filter_models",
        type=str,
        nargs="*",
        default=None,
        help="List of model names to evaluate. If not provided, all models will be evaluated."
    )   
    
    
    for model in models_list:
        
        # logger.info(model)
        
        ckpt_file = model["ckpt_file"]
        model_name = model["model_name"]
        pretrain_data = model["pretrain_data"]
        finetune_data = 'seam_ai'
        
        # "model_name": model_dir.name,
        # "combination": int(repetition_dir),
        # "pretrain_data": pretrain_data,
        # "input_size": int(input_size),
        # "batch_size": int(batch_size),
        # "epochs": int(epochs),
        # "learning_rate": float(learning_rate),
        # "epoch_save": epoch_save,
        # "epoch_trained": int(ckpt_epoch),
        
        

        data_path_mapping = {
        'seam_ai_N':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai_N',
        'seam_ai':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai',
        'f3':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation',
        'f3_N':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation_N',
        }

        data_path = data_path_mapping[finetune_data]
        
        logger.info(f'Checkpoint file: {ckpt_file}')
        logger.info(f'Backbone loaded: {pretrain_data}')
        logger.info(f'Data Path :{data_path}')
    
        
        
        # Rodar avaliação
        
        main(
            ckpt_file=ckpt_file,
            model_name=model_name,
            finetune_data=finetune_data,
            pretrain_data=pretrain_data,
            data_path=data_path,
            num_epochs=50,
            batch_size=8,
            repetition=args.combination,
            ckpt_path=TEST_CKPT_PATH,
            logs_path=TEST_LOGS_PATH,
            gpus=args.gpus,
        )