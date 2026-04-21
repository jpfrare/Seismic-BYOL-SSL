from torchvision.datasets import ImageFolder
import numpy as np
import os
from pathlib import Path
from PIL import Image

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

#a diferença da estrutura da validação é um tanto diferente, funciona como se fosse um dicionário
# Validação: acesso por nomes de colunas (dtype names)
# linha['class_index'] -> Label oficial (0-999) - USAR ESTE PARA O LOSS
# linha['actual_index'] -> ID sequencial da imagem (1-50000)
class ImagenetValReader:
    def __init__(self, root, entries_path):
        self.root = Path(root)
        self.data = np.load(entries_path, allow_pickle=True)
        # 'actual_index' é o que o seu dtype confirmou que existe
        self.targets = [int(row['class_index']) for row in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Como não existe a coluna 'file_name', montamos o nome padrão:
        # ILSVRC2012_val_00000001.JPEG, etc.
        img_name = f"ILSVRC2012_val_{idx+1:08d}.JPEG"
        img_path = self.root / img_name
        
        # Pega a label pela coluna confirmada
        label = int(self.data[idx]['class_index'])
        
        img = Image.open(img_path).convert("RGB")
        return img, label