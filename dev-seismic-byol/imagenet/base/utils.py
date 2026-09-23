from scipy.io import loadmat
import os
import torch
import numpy as np

def get_state_dict(model):
        # O state_dict é um dicionário que representa o estado atual do modelo,
    # contendo seus parâmetros treináveis (pesos e bias) e buffers internos
    # (como estatísticas de BatchNorm).
    #
    # Ele NÃO inclui informações como época, estado do otimizador ou scheduler.
    # Esses elementos fazem parte de um checkpoint completo de treinamento.
    #
    # Aqui, os nomes das chaves são ajustados para incluir o prefixo "RN50model.",
    # garantindo compatibilidade com outros state_dicts ou arquiteturas esperadas.

    state_dict = model.state_dict()
    renamed_state_dict = {}
    for key in state_dict.keys():
        # Replace the key prefix to match my_state_dict
        new_key = f"RN50model.{key}" if not key.startswith("RN50model.") else key
        renamed_state_dict[new_key] = state_dict[key]

    return renamed_state_dict    

def get_dataset_mapping():
    
    nodename = os.uname().nodename
    
    if 'sdumont' in nodename:
        dataset_mapping = {
            'seam_ai_N':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/tiff_data/seam_ai_N',
            'seam_ai':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/tiff_data/seam_ai',
            'f3':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/tiff_data/f3_segmentation',
            'f3_N':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/tiff_data/f3_segmentation_N',
            'both':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/tiff_data/both',
            'both_N':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/tiff_data/both_N',
            'a700':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/a700',
            'namss':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/NAMSS/Data/NAMSS/patch_512_0',
        }
    
    elif '4be00dc0e281' in nodename:
        dataset_mapping = {
            'seam_ai_N':'/workspaces/shared_data/seam_ai_datasets/seam_ai_N/images',
            'seam_ai':'/workspaces/shared_data/seam_ai_datasets/seam_ai/images',
            'f3':'/workspaces/shared_data/seismic/f3_segmentation/images',
            'f3_N':'/workspaces/shared_data/seismic/f3_segmentation_N/images',
            'both':'/workspaces/shared_data/seismic/both/images',
            'both_N':'/workspaces/shared_data/seismic/both_N/images',
        }
        
    elif 'node' in nodename:
        dataset_mapping = {
            'seam_ai_N':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai_N/images',
            'seam_ai':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai/images',
            'f3':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation/images',
            'f3_N':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation_N/images',
            'both':'/home/vinicius.soares/asml/datasets/tiff_data/both/images',
            'both_N':'/home/vinicius.soares/asml/datasets/tiff_data/both_N/images',
            'a700':'/parceirosbr/asml/datasets/a700',
        }
    else:
        raise RuntimeError(f"Unsupported nodename '{nodename}'. Unable to determine dataset mapping.")
    
    return dataset_mapping


def check_transfer_learning(missing_keys):
    missing_resnet_layers = [
    key for key in missing_keys 
    if any(layer_name in key for layer_name in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4'])
    ]

    if len(missing_resnet_layers) == 0:
        print(" [SUCESSO ABSOLUTO] Todas as camadas da ResNet50 foram transferidas e estão tranquilas!")
    else:
        print(" [ALERTA] Algumas camadas da ResNet não foram preenchidas:")
        for layer in missing_resnet_layers[:10]:
            print(f"   └─> Falhou: {layer}")
        if len(missing_resnet_layers) > 10:
            print(f"   └─> ... e mais {len(missing_resnet_layers) - 10} camadas.")
        raise RuntimeError("Mismatch de chaves no Transfer Learning! O esqueleto da ResNet ficou vazio.")

        





