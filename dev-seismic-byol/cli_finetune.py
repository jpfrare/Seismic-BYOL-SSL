import argparse
from train_freeze import main
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
    
    
def parse_freeze_list(values):
    # Se o argparse passou como lista (nargs=5)
    if isinstance(values, list):
        raw_list = values
    else:
        # Se veio como string única
        raw_list = values.replace(",", " ").split()

    try:
        lst = [int(v) for v in raw_list]
        if len(lst) != 5:
            raise argparse.ArgumentTypeError("freeze_list must contain exactly 5 values (0 or 1).")
        if any(v not in (0, 1) for v in lst):
            raise argparse.ArgumentTypeError("freeze_list values must be either 0 or 1.")
        return [bool(v) for v in lst]  # converte 0/1 → False/True
    except ValueError:
        raise argparse.ArgumentTypeError("freeze_list must be a list of integers (0 or 1).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tuning script for seismic segmentation using pretrained BYOL backbone."
    )

    parser.add_argument(
        "--pretrain_data",
        type=str,
        required=True,
        default="seam_ai_N",
        help="Dataset used in pretraining (e.g., f3, seam_ai, both)",
    )
    parser.add_argument(
        "--finetune_data",
        type=str,
        required=True,
        default="seg",
        help="Dataset used for fine-tuning (e.g., f3, seam_ai, both)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--repetition", type=int, default=0, help="Experiment repetition index"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--cap",
        type=parse_cap,
        default=10,
        help="Fraction of data to use for training (between 0 and 1)",
    )
    parser.add_argument(
        "--freeze", action="store_true", help="Whether to freeze the encoder backbone"
    )
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=[0], help="List of GPU indices to use"
    )
    parser.add_argument(
        "--linear", action="store_true", help="If true uses a linear prediction head"
    )
    parser.add_argument(
        "--steps", action="store_true", help="Defines if num_epochs refers to train steps"
    )
    parser.add_argument(
        "--freeze_list",
        type=parse_freeze_list,
        default=None,
        help="Binary list of five values to freeze specific blocks (0 = train, 1 = freeze). "
            "Example: --freeze_list 1 1 1 0 0 OR --freeze_list '1,1,1,0,0'"
    )

    args = parser.parse_args()

    root = '/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/checkpoints'

    PRETRAIN_LOGS_PATH = f"{root}/logs_vinicius/train_freeze_modules/{args.repetition}" if not args.linear else f"{root}/logs_vinicius/train_linear_freeze_modules/{args.repetition}"
    PRETRAIN_CKPT_PATH = f"{root}/ckpt_vinicius/train_freeze_modules/{args.repetition}" if not args.linear else f"{root}/ckpt_vinicius/train_linear_freeze_modules/{args.repetition}"
    # IMPORT_ROOT_PATH = f"/petrobr/parceirosbr/home/vinicius.soares/workspace/Seismic-Byol/dev-seismic-byol/ckpt/pretrain"
    IMPORT_ROOT_PATH = f"/petrobr/parceirosbr/home/vinicius.soares/workspace/Seismic-Byol/dev-seismic-byol/checkpoints/ckpt_vinicius/pretrain"

    dataset_mapping = get_dataset_mapping()
    
    finetune_list = [
        "f3",
        "f3_N",
        "seam_ai",
        "seam_ai_N"
    ]

    if args.finetune_data not in finetune_list:
        raise KeyError(
            f"Dataset '{args.finetune_data}' not found in available options: {finetune_list}"
        )

    pretrain_list = [
        "f3",
        "f3_N",
        "seam_ai",
        "seam_ai_N",
        "both",
        "both_N",
        "s0",
        "a700",
        "imagenet",
        "coco",
        "sup",
        "seg",
        "namss",
    ]

    if args.pretrain_data not in pretrain_list:
        raise KeyError(
            f"Pretrain '{args.pretrain_data}' not found in available options: {pretrain_list}"
        )

    # Simulação de treinamento
    logger.info(" =-=-=- Beginning fine-tuning =-=-=-")
    logger.info(f"Pretrain data: {args.pretrain_data}")
    logger.info(f"Finetune data: {args.finetune_data}")
    logger.info(f"Batch size: {args.batch_size}, LR: {args.learning_rate}")
    logger.info(f"Cap: {args.cap}, Freeze: {args.freeze}")
    logger.info(f"Cap type: {type(args.cap)}")
    logger.info(f"Prediction head: {'linear' if args.linear == True else 'DLV3'}")
    if args.freeze_list:
        logger.info(f"Freeze mask: {args.freeze_list}")

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
        linear=args.linear,
        steps=args.steps,
        modules_frozen=args.freeze_list,   # <--- Novo parâmetro

    )
