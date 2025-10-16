from pathlib import Path
import re
import pandas as pd
from io import StringIO


def extract_epoch_number(filename):
    match = re.search(r"epoch=(\d+)", filename)
    return int(match.group(1)) if match else -1

def extract_step_number(filename):
    match = re.search(r"step=(\d+)", filename)
    return int(match.group(1)) if match else -1


def get_models_files(base_dir="./checkpoints/ckpt_vinicius/pretrain", target_repetition=None):
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

            # Fix: regex supports scientific notation correctly now
            match = re.match(
                r"V(\d+)_pretrain_(.+?)_In(\d+)_B(\d+)_E(\d+)_lr([\deE\.-]+)", model_dir.name
            )
            if not match:
                continue

            _, pretrain_data, input_size, batch_size, epochs, learning_rate = match.groups()

            for train_data_dir in model_dir.iterdir():
                if not train_data_dir.is_dir():
                    continue

                ckpt_files = [
                    f for f in train_data_dir.iterdir()
                    if f.is_file() and re.match(r"step=\d+\.ckpt", f.name)
                ]
                # ckpt_files = [
                #     f for f in train_data_dir.iterdir()
                #     if f.is_file() and re.match(r"epoch=\d+\.ckpt", f.name)
                # ]
                for ckpt_file in ckpt_files:
                    epoch_save = extract_step_number(ckpt_file.name)
                    results.append(
                        {
                            "model_name": model_dir.name,
                            "repetition": int(repetition_dir),
                            "pretrain_data": pretrain_data,
                            "input_size": int(input_size),
                            "batch_size": int(batch_size),
                            "epochs": int(epochs),
                            "learning_rate": float(learning_rate),
                            "ckpt_file": str(ckpt_file),
                            "epoch_save": epoch_save,
                        }
                    )

    return results


# === Load model files and filter by pretrain data ===
df = pd.DataFrame(get_models_files(base_dir="./checkpoints/ckpt_vinicius/pretrain/"))


# filtered_df = df[df["pretrain_data"] == "namss"]
filtered_df = df[df["repetition"] == 4]
filtered_df = filtered_df[filtered_df['epochs'] == 125000]

print(filtered_df)

print(f"Tamanho do DataFrame filtered_df: {len(filtered_df)}")

combination_df = pd.read_csv('temp_df.csv')

# print(combination_df)

# 🧼 Limpa os nomes das colunas
combination_df.columns = combination_df.columns.str.strip()

# Agora a conversão de tipos deve funcionar
combination_df["learning_rate"] = combination_df["learning_rate"].astype(float)
combination_df["batch_size"] = combination_df["batch_size"].astype(int)
combination_df["input_size"] = combination_df["input_size"].astype(int)
combination_df["repetition"] = combination_df["repetition"].astype(int)

# Ensure both DataFrames have matching dtypes for merge keys
merge_keys = ["learning_rate", "batch_size", "input_size", "repetition"]
for key in merge_keys:
    filtered_df.loc[:, key] = pd.to_numeric(filtered_df[key], errors="coerce")
    combination_df[key] = pd.to_numeric(combination_df[key], errors="coerce")

# Filter filtered_df to keep only rows with combinations present in combination_df
df_filtered_combinations = filtered_df.merge(
    combination_df,
    on=merge_keys,
    how="left"
)

# Save for inspection (optional)
df_filtered_combinations.to_csv("df_filtered_combinations.csv", index=False)

# combination_df.to_csv("temp_df_2.csv", index=False)
