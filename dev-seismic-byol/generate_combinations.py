import pandas as pd
import re
from collections import defaultdict
import os

# Diretórios extraídos manualmente da imagem (simulação)
# Listar todos os diretórios em ht_ckpt/pretrain
base_path = "ckpt_ht/pretrain"
directories = []
for d in os.listdir(base_path):
    sub_path = os.path.join(base_path, d)
    if os.path.isdir(sub_path):
        sub_dirs = [sd for sd in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, sd))]
        directories.extend(sub_dirs)
directories = sorted(directories)  # Ordenar alfabeticamente
print(directories)

# Armazenar os dados extraídos
data = []
combo_map = {}
combo_counter = 0

for d in directories:
    match = re.match(r"V(\d+)_pretrain_seam_ai_In(\d+)_B(\d+)_E\d+_lr([\de\.-]+)", d)
    if match:
        repetition, input_size, batch_size, lr = match.groups()
        repetition = int(repetition)
        input_size = int(input_size)
        batch_size = int(batch_size)
        learning_rate = float(lr)

        combo_key = (batch_size, input_size, learning_rate)
        if combo_key not in combo_map:
            combo_map[combo_key] = combo_counter
            combo_counter += 1

        data.append({
            "batch_size": batch_size,
            "input_size": input_size,
            "learning_rate": learning_rate,
            "repetition": repetition,
            "combination": combo_map[combo_key]
        })

# Criar DataFrame
df = pd.DataFrame(data)
print(df)
csv_path = "pretrain_combinations.csv"
df.to_csv(csv_path, index=False)