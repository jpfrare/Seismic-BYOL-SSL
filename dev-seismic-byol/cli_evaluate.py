import argparse
from evaluate import main  # seu main de avaliação
from functions import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for seismic segmentation models."
    )
    parser.add_argument(
        "--repetition",
        type=int,
        required=True,
        help="Experiment repetition index to evaluate."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of epochs (needed for Trainer even in evaluation mode)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="List of GPU indices to use."
    )
    parser.add_argument(
        "--linear", action="store_true", help="If true uses a linear prediction head"
    )

    args = parser.parse_args()

    TEST_LOGS_PATH = f"logs/test/{args.repetition}"
    TEST_CKPT_PATH = f"ckpt/test/{args.repetition}"
    
    logger.info(f"Target repetition: {args.repetition}")
    
    models_list = get_models_files(target_repetition=args.repetition)
    
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
        
        logger.info(model)
        
        model_cap = model["cap"]        
        ckpt_file = model["ckpt_file"]
        model_name = model["model_name"]
        pretrain_data = model["pretrain_data"]
        finetune_data = model["train_data"]

        data_path_mapping = {
            'seam_ai_N':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai_N',
            'seam_ai':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai',
            'f3':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation',
            'f3_N':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation_N',
        }

        data_path = data_path_mapping[finetune_data]

        logger.info(f"Backbone loaded: {pretrain_data}")
        logger.info(f"Data Path :{data_path}")

        # Rodar avaliação

        main(
            ckpt_file=ckpt_file,
            model_name=model_name,
            finetune_data=finetune_data,
            pretrain_data=pretrain_data,
            data_path=data_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            repetition=args.repetition,
            ckpt_path=TEST_CKPT_PATH,
            logs_path=TEST_LOGS_PATH,
            gpus=args.gpus,
            linear=args.linear,
        )
