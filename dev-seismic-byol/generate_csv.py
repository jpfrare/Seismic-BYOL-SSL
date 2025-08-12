import yaml
import csv
from pathlib import Path
import re
import argparse

def extract_metrics_from_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    classification = data.get("classification", {})
    return {
        "acc": classification.get("acc", [None])[0],
        "f1_weighted": classification.get("f1-weighted", [None])[0],
        "mIoU": classification.get("mIoU", [None])[0]
    }

def extract_model_metadata(model_name):

    # match = re.match(r"V(\d+)_pretrain_(.+?)_In(\d+)_B(\d+)_E(\d+)_lr([\deE\.-]+)_step_(\d+)", model_name)
    match = re.match(r"V(\d+)_pre_(.+?)_train_(.+?)_cap_(.+)", model_name)
    if not match:
        raise ValueError(f"Invalid model name format: {model_name}")
    
    repetition, pretrain_data, finetune_data, cap_raw = match.groups()

    # Determine cap_type and clean cap value
    if cap_raw.endswith("_img"):
        cap_type = "images"
        cap_value = cap_raw.replace("_img", "")
    elif cap_raw.endswith("%"):
        cap_type = 'percentage'
        if finetune_data == 'f3' or finetune_data == 'f3_N':
            cap_value = 992
        elif finetune_data == 'seam_ai' or finetune_data == 'seam_ai_N':
            cap_value = 1121
        else:
            cap_value = cap_raw.replace("%", "")
    else:
        cap_type = "unknown"
        cap_value = cap_raw  # fallback

    return {
        "model_name": model_name,
        "repetition": repetition,
        "pretrain_data": pretrain_data,
        "finetune_data": finetune_data,
        "cap": cap_value
    }

def load_existing_csv(csv_path):
    if not csv_path.exists():
        return {}

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        return {row["model_name"]: row for row in reader}

def save_to_csv(rows_dict, output_csv):
    rows = list(rows_dict.values())
    fieldnames = [
        "model_name",
        "repetition",
        "pretrain_data",
        "finetune_data",
        "cap", 
        "acc", 
        "f1_weighted", 
        "mIoU"
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def collect_metrics_to_csv(logs_root, repetition, output_csv):
    metrics_dir = Path(logs_root) / str(repetition)
    rows_dict = load_existing_csv(Path(output_csv))

    if not metrics_dir.exists():
        raise FileNotFoundError(f"No directory found at {metrics_dir}")

    new_entries = 0
    updated_entries = 0

    for model_dir in metrics_dir.iterdir():
        if not model_dir.is_dir():
            continue

        metadata = extract_model_metadata(model_dir.name)
        metadata['repetition'] = repetition

        for yaml_file in model_dir.glob("metrics_*.yaml"):
            metrics = extract_metrics_from_yaml(yaml_file)
            row = {
            **metadata,
            **metrics
            }
            model_name = metadata["model_name"]
            if model_name in rows_dict:
                rows_dict[model_name].update(row)
                updated_entries += 1
                # print(model_name)
            else:
                rows_dict[model_name] = row
                new_entries += 1

    if new_entries + updated_entries == 0:
        raise RuntimeError(f"No metric files found in {metrics_dir}")

    save_to_csv(rows_dict, Path(output_csv))

    print(f"✅ {new_entries} new, {updated_entries} updated rows written to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export evaluation metrics to CSV (append/update mode)")
    # parser.add_argument("--repetition", type=int, required=True, help="Repetition number")
    parser.add_argument("--logs_root", type=str, default="logs_ht/test_02_unfreeze_rerun", help="Root logs/test directory")
    parser.add_argument("--output_csv", type=str, default="ht_eval_linear_unfreeze.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    
    list_of_combinations = [46]
    
    
    for number in list_of_combinations:

    collect_metrics_to_csv(
        logs_root=args.logs_root,
        repetition=args.repetition,
        output_csv=args.output_csv
        )
