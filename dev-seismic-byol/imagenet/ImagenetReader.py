from torchvision.datasets import ImageFolder
import numpy as np
import os
from pathlib import Path
from PIL import Image


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

class ImagenetValReader:
    def __init__(self, root, entries_path):
        self.root = Path(root)
        # Carrega o array estruturado
        self.data = np.load(entries_path, allow_pickle=True)
        
        # Lista as imagens exatamente como o teste do terminal fez
        self.image_files = sorted([
            f for f in os.listdir(root) if f.endswith('.JPEG')
        ])
        
        if len(self.image_files) != len(self.data):
            raise ValueError(
                f"Mismatch: {len(self.image_files)} imagens no disco vs "
                f"{len(self.data)} entradas no .npy"
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Pega o nome do arquivo pela ordem alfabética
        img_name = self.image_files[idx]
        img_path = self.root / img_name
        
        # Pega o label usando o nome da coluna que o terminal nos deu
        # Usamos 'actual_index' como o ID da classe para o PyTorch
        label = int(self.data[idx]['actual_index'])
        
        # Abre a imagem
        img = Image.open(img_path).convert("RGB")
        
        return img, label