import numpy as np
from pathlib import Path
from PIL import Image
import os
from ImagenetReader import *

# --- CONFIGURAÇÃO DE CAMINHOS ---
# Ajuste estes caminhos para a realidade do seu ambiente no SDumont
BASE_DIR = Path("/petrobr/parceirosbr/spfm/datasets/ImageNet_2012")
TRAIN_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train"
VAL_ROOT = VAL_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val"
TRAIN_NPY = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy"
GT_TXT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
META_MAT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/meta.mat"

def test_full_dataset():
    print("=== Iniciando Teste de Integração: Treino vs Validação ===\n")

    # 1. Instanciar os leitores
    try:
        reader_train = ImagenetReader(TRAIN_ROOT, TRAIN_NPY)
        reader_val = ImagenetValReader(VAL_ROOT, GT_TXT, META_MAT)
        print(f"✓ Readers instanciados com sucesso.")
        print(f"✓ Treino: {len(reader_train)} imagens | Validação: {len(reader_val)} imagens")
    except Exception as e:
        print(f"✗ Erro na instanciação: {e}")
        return

    # 2. Teste de Sanidade: Amostras de Treino
    print("\n--- Verificando Treino (Amostras Aleatórias) ---")
    for i in [0, len(reader_train) // 2, len(reader_train) - 1]:
        img, label = reader_train[i]
        print(f"Index {i:7d} | Label: {label:3d} | Size: {img.size}")
    
    # 3. Teste de Sanidade: Amostras de Validação
    print("\n--- Verificando Validação (Amostras Aleatórias) ---")
    # Testando a primeira imagem (que você sabe que é ID 490 no TXT)
    for i in [0, 100, 49999]:
        img, label = reader_val[i]
        print(f"Val Index {i:5d} | Label Traduzida: {label:3d} | Size: {img.size}")

    # 4. O TESTE FINAL: SINCRONIA DE LABELS
    # Vamos verificar se a label '0' no treino e na validação apontam para o mesmo WNID
    print("\n--- Verificação de Sincronia de Labels ---")
    
    # No treino, pegamos o WNID da label 0 direto do dado estruturado
    train_sample_label_0 = reader_train.data[0]
    wnid_label_0_train = train_sample_label_0[2]
    
    # Na validação, precisamos ver qual WNID o mapeador atribuiu à label 0
    # Invertendo o dicionário do reader_val para conferir
    idx_to_id = {v: k for k, v in reader_val.id_to_label.items()}
    id_comp_label_0 = idx_to_id[0]
    
    # Usando o synset_words.txt ou meta.mat para ver qual WNID é o ID da competição
    from scipy.io import loadmat
    meta = loadmat(META_MAT)['synsets']
    wnid_label_0_val = next(str(m[0][1][0]) for m in meta if int(m[0][0][0][0]) == id_comp_label_0)

    print(f"Classe 0 no Treino (via .npy): {wnid_label_0_train}")
    print(f"Classe 0 na Validação (via meta.mat): {wnid_label_0_val}")

    if wnid_label_0_train == wnid_label_0_val:
        print("\n🔥 SUCESSO TOTAL: As labels estão sincronizadas!")
        print("Agora 0 significa a mesma coisa em ambos os datasets.")
    else:
        print("\n⚠️ ALERTA: As labels ainda estão desalinhadas.")
        print("Verifique se a ordem alfabética do treino é a mesma usada no mapeamento.")

if __name__ == "__main__":
    # Certifique-se que o scipy está instalado: pip install scipy
    test_full_dataset()