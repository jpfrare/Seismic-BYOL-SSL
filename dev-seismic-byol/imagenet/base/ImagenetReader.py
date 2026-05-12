from torchvision.datasets import ImageFolder
import numpy as np
import os
from pathlib import Path
from PIL import Image
from scipy.io import loadmat

#por alguma razão mistica, tanto o dataset de validação como de treino sao arrays estruturados do numpy
#pra cada linha do array do treino:
#linha[0] é o id do arquivo ex: 10026
#linha[1] é a label dele
#linha[2] é o prefixo/subpasta ex: n01440764
#você consegue fazer o caminho da imagem juntando essas informações e a label obviamente está na linha[1], ai você forma uma sample
class ImagenetReader:
    def __init__(self, root, entries_path):
        self.root = Path(root)
        # Carrega o array estruturado 
        self.data = np.load(entries_path, allow_pickle=True)
        
        # Mapeia as labels (coluna index 1) para o StratifiedSubset
        # Fazemos isso no init para o subset não precisar iterar depois
        self.targets = [int(row[1]) for row in self.data]

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data[idx]
        img_idx = row[0]
        label = int(row[1])
        wnid = row[2]
        
        # Reconstrói o caminho validado no teste: wnid/wnid_idx.JPEG
        img_path = self.root / wnid / f"{wnid}_{img_idx}.JPEG"
        
        # Abre a imagem de forma segura
        img = Image.open(img_path).convert("RGB")
        return img, label

class ImagenetValReader():
    def __init__(self, root, gt_path, meta_path):
        self.root = Path(root)
        
        # 1. Mapeamento ID TXT -> WNID -> Índice Alfabético (Treino)
        meta = loadmat(meta_path)['synsets']
        id_to_wnid = {int(m[0][0][0][0]): str(m[0][1][0]) for m in meta[:1000]}
        
        all_wnids = sorted([str(m[0][1][0]) for m in meta[:1000]])
        wnid_to_idx = {wnid: i for i, wnid in enumerate(all_wnids)}
        
        self.id_to_label = {comp_id: wnid_to_idx[wnid] for comp_id, wnid in id_to_wnid.items()}
        
        # 2. Carrega labels do ground_truth.txt
        with open(gt_path, 'r') as f:
            raw_ids = [int(line.strip()) for line in f.readlines()]
            
        self.targets = [self.id_to_label[id_bruto] for id_bruto in raw_ids]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Localiza a imagem (ILSVRC2012_val_00000001.JPEG)
        img_name = f"ILSVRC2012_val_{idx+1:08d}.JPEG"
        img_path = self.root / img_name
        
        # Retorna a imagem bruta (PIL) e a label traduzida
        img = Image.open(img_path).convert("RGB")
        label = self.targets[idx]
        
        return img, label