# cli.py

import argparse
from pretrain import main

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
        "seam_ai_N": "/home/vinicius.soares/asml/datasets/tiff_data/seam_ai_N/images",
        "seam_ai": "/home/vinicius.soares/asml/datasets/tiff_data/seam_ai/images",
        "f3": "/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation/images",
        "f3_N": "/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation_N/images",
        "both": "/home/vinicius.soares/asml/datasets/tiff_data/both/images",
        "both_N": "/home/vinicius.soares/asml/datasets/tiff_data/both_N/images",
        "s0": "/home/vinicius.soares/asml/datasets/S0.M.B316.P3415.E0302.LALL.PSDM_MTVFL.2013.01.P.zarr",
        "a700": "/home/vinicius.soares/asml/datasets/a700",
    }

    if args.dataset_name not in dataset_mapping.keys():
        raise KeyError(
            f"Dataset '{args.dataset_name}' not found in available options: {list(dataset_mapping.keys())}"
        )

    PRETRAIN_LOGS_PATH = f"logs/pretrain/{args.repetition}"
    PRETRAIN_CKPT_PATH = f"ckpt/pretrain/{args.repetition}"

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
