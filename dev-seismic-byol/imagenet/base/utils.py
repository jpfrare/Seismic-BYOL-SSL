from scipy.io import loadmat

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


def build_son_to_father_dict(father_id, sysnets, dic):
    infos = sysnets[father_id - 1][0]
    num_children = int(infos[4][0][0])

    if num_children == 0:
        return
    
    for children_id in infos[5][0]:
        int_children_id = int(children_id)
        dic[int_children_id] = father_id
        build_son_to_father_dict(int_children_id, sysnets, dic)
    
def build_heritage_path(mat_path):
    mat = loadmat(mat_path)
    synsets = mat['synsets']

    dic = {}
    build_son_to_father_dict(1001, synsets, dic)

    heritage_path = {} #wnid -> lista do wnid de todos os parentes começando por ele mesmo até a classe entidade

    for i in range(0,1000):
        imagenet_id = i + 1
        wnid = str(synsets[i][0][1][0])

        heritage_path[wnid] = [wnid]
        
        current_id = imagenet_id

        while current_id in dic:
            father_id = dic[current_id]
            father_wnid = str(synsets[father_id - 1][0][1][0])
            heritage_path[wnid].append(father_wnid)

            current_id = father_id

    
    return heritage_path


def reduce_taxonomic_diversity(wnids: list, top_down: bool, level: int, mat_path: str) -> (dict, int):
    wnid_to_class = {}
    coarse_to_class = {}
    current_class_id = 0

    heritage_path = build_heritage_path(mat_path)

    for wnid in wnids:
        ancestors = heritage_path[wnid] #lista de todos os wnids antepassados até entidade (o último elemento é a classe entidade)

        if top_down:
            pos = len(ancestors) - 1 - level
            chosen_wnid = ancestors[pos] if pos > 0 else ancestors[0]
        else:
            chosen_wnid = ancestors[level] if level < len(ancestors) else ancestors[3]
        
        if chosen_wnid not in coarse_to_class:
            coarse_to_class[chosen_wnid] = current_class_id
            current_class_id += 1
        
        wnid_to_class[wnid] = coarse_to_class[chosen_wnid]

    
    return (wnid_to_class, current_class_id)

