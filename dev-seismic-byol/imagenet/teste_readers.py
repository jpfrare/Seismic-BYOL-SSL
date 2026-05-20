import numpy as np
from scipy.io import loadmat
from pathlib import Path

# Paths do seu ambiente
TRAIN_ENTRIES = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy"
MAT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/meta.mat"

# 1. Simula a lógica de labels da validação (Ordem Alfabética)
meta = loadmat(MAT_ROOT)['synsets']
all_wnids_val = sorted([str(m[0][1][0]) for m in meta[:1000]])
wnid_to_label_val = {wnid: i for i, wnid in enumerate(all_wnids_val)}

# 2. Carrega uma amostra do treino
train_data = np.load(TRAIN_ENTRIES, allow_pickle=True)

print("=== VERIFICAÇÃO DE ALINHAMENTO DE LABELS ===")
desalinhados = 0
amostras_para_mostrar = 5

for i in range(min(500, len(train_data))):
    row = train_data[i]
    label_treino = int(row[1])
    wnid_treino = row[2]
    
    # Qual label a validação daria para esse mesmo WNID?
    label_val_esperada = wnid_to_label_val[wnid_treino]
    
    if label_treino != label_val_esperada:
        desalinhados += 1
        if amostras_para_mostrar > 0:
            print(f"Erro no WNID {wnid_treino}: No treino a label é [{label_treino}], mas na validação mapeia para [{label_val_esperada}]")
            amostras_para_mostrar -= 1

print("--------------------------------------------")
print(f"Total de amostras testadas: {i+1}")
print(f"Quantidade de mapeamentos desalinhados: {desalinhados}")
if desalinhados > 0:
    print("❌ CRÍTICO: Seus Readers estão com as labels desalinhadas!")
else:
    print("✅ Sucesso: As labels estão alinhadas.")