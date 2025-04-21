# cli.py

import argparse
from pretrain import main
from functions import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretraining script for seismic segmentation with BYOL."
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input size (used for both height and width)",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="seam_ai", help="Name of the dataset"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--repetition", type=int, default=0, help="Repetition being run"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.2, help="Models learning rate"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="List of GPU device indices to use",
    )

    args = parser.parse_args()

    dataset_mapping = {
        "seam_ai_N": "/workspaces/shared_data/seam_ai_datasets/seam_ai_N/images",
        "seam_ai": "/workspaces/shared_data/seam_ai_datasets/seam_ai/images",
        "f3": "/workspaces/shared_data/seismic/f3_segmentation/images",
        "f3_N": "/workspaces/shared_data/seismic/f3_segmentation_N/images",
        "both": "/workspaces/shared_data/seismic/both/images",
        "both_N": "/workspaces/shared_data/seismic/both_N/images",
    }

    logger.info(" =-=-=- Begining training =-=-=-")

    logger.info(f"Dataset path: {dataset_mapping[args.dataset_name]}")

    if args.dataset_name not in dataset_mapping.keys():
        logger.error("Dataset not available")
        raise KeyError(
            f"Dataset '{args.dataset_name}' not found in available options: {list(dataset_mapping.keys())}"
        )

    PRETRAIN_LOGS_PATH = f"logs/pretrain/{args.repetition}"
    PRETRAIN_CKPT_PATH = f"ckpt/pretrain/{args.repetition}"

    logger.info(f"Batches: {args.batch_size} - Input: {args.input_size}")
    logger.info(f"Pretrain Log Path: {PRETRAIN_LOGS_PATH}")
    logger.info(f"Pretrain CKPT Path: {PRETRAIN_CKPT_PATH}")

    main(
        input_size=(args.input_size, args.input_size),
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        repetition=args.repetition,
        learning_rate=args.learning_rate,
        data_path=dataset_mapping[args.dataset_name],
        log_path=PRETRAIN_LOGS_PATH,
        ckpt_path=PRETRAIN_CKPT_PATH,
        gpus=args.gpus,
    )
