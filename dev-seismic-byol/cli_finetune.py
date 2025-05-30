import argparse
from train import main
from functions import *

def parse_cap(value):
    try:
        if '.' in value:
            val = float(value)
            if not (0.0 < val <= 1.0):
                raise argparse.ArgumentTypeError("Float cap must be in the (0, 1] range.")
            return val
        else:
            val = int(value)
            if val <= 0:
                raise argparse.ArgumentTypeError("Integer cap must be greater than 0.")
            return val
    except ValueError:
        raise argparse.ArgumentTypeError("Cap must be a float in (0, 1] or a positive int.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tuning script for seismic segmentation using pretrained BYOL backbone."
    )

    parser.add_argument(
        "--pretrain_data",
        type=str,
        required=True,
        default="f3",
        help="Dataset used in pretraining (e.g., f3, seam_ai, both)"
    )
    parser.add_argument(
        "--finetune_data",
        type=str,
        required=True,
        default="f3",
        help="Dataset used for fine-tuning (e.g., f3, seam_ai, both)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--repetition", type=int, default=0, help="Experiment repetition index"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--cap",
        type=parse_cap,
        default=1.0,
        help="Amount of data to use for training: float (0-1] for fraction, or int > 0 for fixed number of samples"
    )
    parser.add_argument(
        "--freeze", action="store_true", help="Whether to freeze the encoder backbone"
    )
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=[0], help="List of GPU indices to use"
    )

    args = parser.parse_args()

    PRETRAIN_LOGS_PATH = f"logs/train/{args.repetition}"
    PRETRAIN_CKPT_PATH = f"ckpt/train/{args.repetition}"
    IMPORT_ROOT_PATH = f"ckpt/pretrain/"

    dataset_mapping = {
    'seam_ai_N':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai_N',
    'seam_ai':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai',
    'f3':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation',
    'f3_N':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation_N',
    }

    if args.finetune_data not in dataset_mapping.keys():
        raise KeyError(f"Dataset '{args.finetune_data}' not found in available options: {list(dataset_mapping.keys())}")

    pretrain_list = [
        "f3", "f3_N", "seam_ai", "seam_ai_N", "both", "both_N", "s0", "a700", "imagenet", "coco", "sup",
    ]

    if args.pretrain_data not in pretrain_list:
        raise KeyError(f"Pretrain '{args.pretrain_data}' not found in available options: {pretrain_list}")

    # Simulação de treinamento
    logger.info(" =-=-=- Beginning fine-tuning =-=-=-")
    logger.info(f"Pretrain data: {args.pretrain_data}")
    logger.info(f"Finetune data: {args.finetune_data}")
    logger.info(f"Batch size: {args.batch_size}, LR: {args.learning_rate}")
    logger.info(f"Cap: {args.cap}, Freeze: {args.freeze}")
    logger.info(f"Cap type: {type(args.cap)}")

    # Aqui você chama o método de treinamento
    main(
        pretrain_data=args.pretrain_data,
        finetune_data=args.finetune_data,
        data_path=dataset_mapping[args.finetune_data],
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        repetition=args.repetition,
        learning_rate=args.learning_rate,
        cap=args.cap,
        freeze=args.freeze,
        ckpt_path=PRETRAIN_CKPT_PATH,
        logs_path=PRETRAIN_LOGS_PATH,
        import_root_path=IMPORT_ROOT_PATH,
        gpus=args.gpus,
    )
