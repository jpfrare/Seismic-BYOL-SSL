import os

def retrieve_csvs(base_dir, number):
    result = []
    root_dir = os.path.join(base_dir, str(number))

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".csv"):
                model_name = os.path.basename(os.path.dirname(dirpath))  # one folder above the CSV
                full_path = os.path.join(dirpath, file)
                result.append((model_name, full_path))

    return result

# Example usage:
csvs = retrieve_csvs("path/to/pretrain", 0)
for model_name, csv_path in csvs:
    print(f"{model_name} : {csv_path}")


import csv
from pathlib import Path
import re
import argparse

def extract_model_metadata(model_name):
    # match = re.match(r"V(\d+)_pretrain_(.+?)_In(.+?)_B(.+)_E(.+)", model_name)
    match = re.match(r"V(\d+)_pre_(.+?)_train_(.+?)_(.+?)", model_name)
    if not match:
        raise ValueError(f"Invalid model name format: {model_name}")
    repetition, pretrain_data, train, _= match.groups()
    return {
        "model_name": model_name,
        "repetition": repetition,
        "train_data": train,
        "pretrain_data": pretrain_data,
        }
    
    
import pandas as pd
import matplotlib.pyplot as plt

# base_dir = 'logs/pretrain' 
base_dir = 'logs/train'
csv_paths = [
    item for item in retrieve_csvs(base_dir, 0)
    if extract_model_metadata(item[0])['pretrain_data'] == 'coco'
]
print(len(csv_paths))

# print(csv_paths)  

for item in csv_paths:
    # name = extract_model_metadata(item[0])['pretrain_data']
    name = item[0]
    path = item[1] 

    # df = pd.read_csv(path)

    # # Convert to numeric just in case
    # df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    # df['train_loss'] = pd.to_numeric(df['train_loss'], errors='coerce')

    # # Plot
    # plt.figure(figsize=(8, 4))
    # plt.plot(df['epoch'], df['train_loss'], color='blue', linewidth=1)

    # # Minimal axis formatting
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # plt.xticks(fontsize=8)
    # plt.yticks(fontsize=8)
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title(f'{name}')
    # plt.box(False)
    # plt.savefig(f'outputs/curves/loss_curve_{name}')

    # plt.tight_layout()
    # plt.show()
    # Read your CSV
    df = pd.read_csv(path)  # Replace with actual path

    # Ensure numeric columns
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df['train_loss'] = pd.to_numeric(df['train_loss'], errors='coerce')
    df['val_loss'] = pd.to_numeric(df['val_loss'], errors='coerce')

    # Group by epoch and take the first non-null value (or use mean() if you prefer)
    grouped = df.groupby('epoch').agg({
        'train_loss': 'first',
        'val_loss': 'first'
    }).reset_index()

    # Drop rows where both losses are NaN
    grouped = grouped.dropna(subset=['train_loss', 'val_loss'], how='all')

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(grouped['epoch'], grouped['train_loss'], label='Train Loss', color='blue', linewidth=1)
    plt.plot(grouped['epoch'], grouped['val_loss'], label='Val Loss', color='orange', linewidth=1)

    # Style
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'outputs/curves/loss_curve_{name}')
    plt.close()
