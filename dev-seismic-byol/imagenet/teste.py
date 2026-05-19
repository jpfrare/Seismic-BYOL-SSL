import sys
from pathlib import Path
import numpy as np

# Garante que o Python localize seus arquivos locais no diretório atual
sys.path.append(str(Path(__file__).parent))

from base.ImagenetReader import ImagenetReader, ImagenetValReader

# Caminhos absolutos reais fornecidos do cluster
DATASET_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train"
TRAIN_ENTRIES = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras_v3/entries-TRAIN.npy"
VAL_ROOT = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val"
GT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
MAT_ROOT = "/petrobr/parceirosbr/home/joao.frare/workspace/spfm/sharedata/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/meta.mat"

def test_real_pipeline():
    print("🚀 Iniciando verificação com dados reais do ImageNet no Santos Dumont...")
    
    # 1. Testando Carregamento do Treino (1.2 Milhão de imagens)
    print("\n📂 Instanciando ImagenetReader (Treino)...")
    try:
        train_reader = ImagenetReader(root=DATASET_ROOT, entries_path=TRAIN_ENTRIES)
        print(f"  [OK] Dataset carregado. Total de instâncias: {len(train_reader)}")
        img, label = train_reader[0]
        print(f"  [OK] Leitura física bem-sucedida! Resolução: {img.size} | Label Original: {label}")
    except Exception as e:
        print(f"  ❌ FALHA ao ler dados de treino: {e}")
        return

    # 2. Testando Carregamento da Validação (50k imagens)
    print("\n📂 Instanciando ImagenetValReader (Validação)...")
    try:
        val_reader = ImagenetValReader(root=VAL_ROOT, gt_path=GT_ROOT, meta_path=MAT_ROOT)
        print(f"  [OK] Dataset carregado. Total de instâncias: {len(val_reader)}")
        img_val, label_val = val_reader[0]
        print(f"  [OK] Leitura física bem-sucedida! Resolução: {img_val.size} | Label Inicial: {label_val}")
    except Exception as e:
        print(f"  ❌ FALHA ao ler dados de validação: {e}")
        return

    # 3. Testando a Compressão Taxonômica em Larga Escala
    print("\n🧬 Aplicando colapso de granularidade (Bottom-Up, Level=3)...")
    print("⏳ Consultando WordNet via NLTK (Aguarde alguns segundos)...")
    
    try:
        num_classes_train = train_reader.to_coarse_classes(top_down=False, level=5)
        print(f"  [OK] Treino reduzido para {num_classes_train} superclasses.")
        
        num_classes_val = val_reader.to_coarse_classes(top_down=False, level=5)
        print(f"  [OK] Validação reduzida para {num_classes_val} superclasses.")
        
        # O teste mais importante do seu pipeline científico:
        print("\n⚖️  Validando alinhamento de partições...")
        assert num_classes_train == num_classes_val, (
            f"Divergência detectada! Treino gerou {num_classes_train} classes e Validação gerou {num_classes_val}."
        )
        print(f"  [OK] Excelente! Ambas as partições alinharam perfeitamente em {num_classes_train} classes.")
        
        # Testando __getitem__ pós-mudança taxonômica
        _, label_coarse_train = train_reader[0]
        _, label_coarse_val = val_reader[0]
        print(f"  [OK] Chaveamento dinâmico funcionando! Novas Coarse Labels -> Treino: {label_coarse_train} | Val: {label_coarse_val}")
        
        print("\n🎯 [SUCESSO TOTAL] O pipeline real está consistente, rápido e pronto para o cluster!")
        
    except AssertionError as ae:
        print(f"\n🚨 ERRO CRÍTICO DE ALINHAMENTO: {ae}")
    except Exception as e:
        print(f"\n❌ ERRO NA OPERAÇÃO TAXONÔMICA: {e}")

if __name__ == "__main__":
    test_real_pipeline()