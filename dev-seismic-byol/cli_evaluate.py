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
        default=[],
        help="List of GPU indices to use."
    )
    parser.add_argument(
        "--linear", action="store_true", help="If true uses a linear prediction head"
    )
    
    parser.add_argument(
        "--filter_models",
        type=str,
        nargs="*",
        default=None,
        help="List of model names to evaluate. If not provided, all models will be evaluated."
    )  

    args = parser.parse_args()

    root = '/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/checkpoints'

    TEST_LOGS_PATH = f"{root}/logs_vinicius/test_freeze_modules/{args.repetition}" if not args.linear else f"{root}/logs_vinicius/test_linear_freeze_modules/{args.repetition}"
    TEST_CKPT_PATH = f"{root}/ckpt_vinicius/test_freeze_modules/{args.repetition}" if not args.linear else f"{root}/ckpt_vinicius/test_linear_freeze_modules/{args.repetition}"
    
    
    # PRETRAIN_LOGS_PATH = f"{root}/logs_vinicius/train_freeze_modules/{args.repetition}" if not args.linear else f"{root}/logs_vinicius/train_linear_freeze_modules/{args.repetition}"
    # PRETRAIN_CKPT_PATH = f"{root}/ckpt_vinicius/train_freeze_modules/{args.repetition}" if not args.linear else f"{root}/ckpt_vinicius/train_linear_freeze_modules/{args.repetition}"
    # IMPORT_ROOT_PATH = f"/petrobr/parceirosbr/home/vinicius.soares/workspace/Seismic-Byol/dev-seismic-byol/ckpt/pretrain"

    
    logger.info(f"Target repetition: {args.repetition}")
    
    models_list = get_models_files(
        target_repetition=args.repetition, 
        base_dir=f"{root}/ckpt_vinicius/train_freeze_modules" if not args.linear else f"{root}/ckpt_vinicius/train_freeze_modules")
    
    if args.filter_models:
        models_list = [model for model in models_list if args.filter_models [0] in model["model_name"]]
        
    logger.info(f"Ammount of models found: {len(models_list)}")
    
    # Filter models based on user input
 
    for model in models_list:
        
        logger.info(model)
        
        model_cap = model["cap"]        
        ckpt_file = model["ckpt_file"]
        model_name = model["model_name"]
        pretrain_data = model["pretrain_data"]
        finetune_data = model["train_data"]

        dataset_mapping = get_dataset_mapping()

        # finetune_list = [
        #     "f3",
        #     "f3_N",
        #     "seam_ai",
        #     "seam_ai_N"
        # ]

        # if args.finetune_data not in finetune_list:
        #     raise KeyError(
        #         f"Dataset '{args.finetune_data}' not found in available options: {finetune_list}"
        #     )

        data_path = dataset_mapping[finetune_data]

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
