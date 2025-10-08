import argparse
from train import main
from functions import *
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tuning script for seismic segmentation using pretrained BYOL backbone."
    )
    parser.add_argument(
        "--combination", type=int, default=50, help="Combination to run"
    )
    parser.add_argument(
        "--freeze", action="store_true", help="Whether to freeze the encoder backbone"
    )
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=[0], help="List of GPU indices to use"
    )
    parser.add_argument(
        "--linear", action="store_true", help="Whether to use the linear head"
    )

    args = parser.parse_args()
    
    finetune_data = 'seam_ai_N'
    
    # PRETRAIN_LOGS_PATH = f"ht_logs/train_dlv3/{args.combination}"
    # PRETRAIN_CKPT_PATH = f"ht_ckpt/train_dlv3/{args.combination}"
    # IMPORT_ROOT_PATH = f"ckpt_ht/pretrain/"

    PRETRAIN_LOGS_PATH = f"checkpoints/logs_vinicius/linear_readout_train/{args.combination}"
    PRETRAIN_CKPT_PATH = f"checkpoints/ckpt_vinicius/linear_readout_train/{args.combination}"
    IMPORT_ROOT_PATH = f"checkpoints/ckpt_vinicius/pretrain/"

    # dataset_mapping = {
    # 'seam_ai_N':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai_N',
    # 'seam_ai':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai',
    # 'f3':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation',
    # 'f3_N':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation_N',
    # }
    
    dataset_mapping = get_dataset_mapping()
    
    combination_df = pd.read_csv('temp_df.csv')
    models_df = pd.read_csv('df_filtered_combinations.csv')
    
    target_df = combination_df[combination_df['combination'] == args.combination]
    models_df = models_df[models_df['combination'] == args.combination]
    
    print(models_df)
    
    logger.info(f"Models to be trained: {len(models_df)}")
    
    if args.freeze:
        logger.info("Freezing backbone")
    
    if len(target_df) == 1:
        row_dict = target_df.iloc[0].to_dict()
        logger.info(f"Combination {args.combination}: {row_dict}")

    for _, row in models_df.iterrows():
        row = row.to_dict()
        
        logger.info(f"Import path: {row['ckpt_file']}")
    
        main(
            pretrain_data=finetune_data,
            finetune_data=finetune_data,
            data_path=dataset_mapping[finetune_data],
            num_epochs=50,
            batch_size=8,
            repetition=0,
            learning_rate=0.001,
            cap=1.0,
            freeze=args.freeze,
            ckpt_path=PRETRAIN_CKPT_PATH,
            logs_path=PRETRAIN_LOGS_PATH,
            import_root_path=IMPORT_ROOT_PATH,
            import_path=row['ckpt_file'],
            gpus=args.gpus,
            full_save_name=f'finetune_{row["model_name"]}_step_{row["epoch_save"]}',
            linear=args.linear,
        )
