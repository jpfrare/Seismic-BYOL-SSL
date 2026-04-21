import numpy as np
import os
from pathlib import Path
from PIL import Image

# Configurações de Path (ajuste se necessário)
DATASET_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train"
TRAIN_ENTRIES = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy"

def test_hard_access():
    print("--- TESTE DE ACESSO DIRETO AO TREINO ---")
    
    # 1. Carrega o array estruturado
    data = np.load(TRAIN_ENTRIES, allow_pickle=True)
    total_samples = len(data)
    print(f"Total de entradas no .npy: {total_samples}")

    # 2. Amostras para testar: início, meio (crítico) e fim
    test_indices = [0, total_samples // 2, total_samples - 1]
    
    for idx in test_indices:
        row = data[idx]
        img_idx = row[0]
        label = row[1]
        wnid = row[2]
        
        # Reconstrói o caminho que o Reader vai usar: raiz/wnid/wnid_idx.JPEG
        rel_path = os.path.join(wnid, f"{wnid}_{img_idx}.JPEG")
        full_path = os.path.join(DATASET_ROOT, rel_path)
        
        print(f"\n[Índice {idx}]")
        print(f"  WNID: {wnid} | Label: {label}")
        print(f"  Caminho gerado: {full_path}")
        
        # Teste 1: Existência
        if os.path.exists(full_path):
            print("  ✅ Arquivo encontrado no disco.")
            
            # Teste 2: Permissão de Leitura (Onde o Permission Denied morre)
            try:
                with Image.open(full_path) as img:
                    img.verify() # Verifica se o arquivo não está corrompido
                    print(f"  ✅ Permissão de LEITURA OK. Formato: {img.format}")
            except Exception as e:
                print(f"  ❌ ERRO DE PERMISSÃO/LEITURA: {e}")
        else:
            print("  ❌ FALHA: Arquivo não existe. Verifique se o DATASET_ROOT está correto.")
            # Verificação extra: listar a subpasta do WNID para ver o que tem lá
            subpath = os.path.join(DATASET_ROOT, wnid)
            if os.path.exists(subpath):
                print(f"  ℹ️ A subpasta {wnid} existe. Conteúdo (primeiros 3): {os.listdir(subpath)[:3]}")
            else:
                print(f"  ❌ A subpasta {wnid} sequer existe.")

if __name__ == "__main__":
    test_hard_access()