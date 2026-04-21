import numpy as np
import os
from pathlib import Path

# Configurações - AJUSTE O CAMINHO DA PASTA 'VAL' AQUI
VAL_IMAGES_PATH = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val"
ENTRIES_PATH = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras/entries-VAL.npy"

def test_structure():
    print("--- Inspecionando Estrutura do .npy ---")
    data = np.load(ENTRIES_PATH, allow_pickle=True)
    
    # 1. Tenta ver os nomes das colunas (dtype names)
    if data.dtype.names:
        print(f"Colunas encontradas: {data.dtype.names}")
    else:
        print("O array não tem nomes de colunas, usaremos índices numéricos.")

    # 2. Pega a primeira linha
    first_row = data[0]
    print(f"\nPrimeira linha completa: {first_row}")
    
    # 3. Teste de mapeamento com o disco
    print("\n--- Teste de Mapeamento de Imagem ---")
    # Lista as imagens da pasta real (ordem alfabética é o padrão ImageNet)
    all_imgs = sorted([f for f in os.listdir(VAL_IMAGES_PATH) if f.endswith('.JPEG')])
    
    if len(all_imgs) == 0:
        print(f"ERRO: Nenhuma imagem .JPEG encontrada em {VAL_IMAGES_PATH}")
        return

    print(f"Total de imagens na pasta: {len(all_imgs)}")
    
    # Simula o que o Reader faria
    sample_idx = 0
    img_name = all_imgs[sample_idx]
    label_id = first_row[0] # O 293 que você viu
    wnid = first_row[2]     # O 'n01440764' que você viu
    
    print(f"Índice [{sample_idx}]:")
    print(f"  Arquivo no disco: {img_name}")
    print(f"  Label (ID): {label_id}")
    print(f"  Classe (WNID): {wnid}")
    
    full_path = os.path.join(VAL_IMAGES_PATH, img_name)
    if os.path.exists(full_path):
        print(f"\nSUCESSO: O arquivo {full_path} existe!")
    else:
        print(f"\nFALHA: O arquivo {full_path} não foi encontrado.")

if __name__ == "__main__":
    test_structure()