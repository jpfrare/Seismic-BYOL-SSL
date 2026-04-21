from torchvision.datasets import ImageFolder
import numpy as np
import os
from pathlib import Path
from PIL import Image


class ImagenetReader():
    def __init__(self, root):
        self.ds = ImageFolder(root)
        self.samples = self.ds.samples #uma lista de tuplas  (caminho_imagem, label)
        self.targets = self.ds.targets    #apenas as labels

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.ds.samples[idx]
        img = self.ds.loader(path)      #carregando imagem pra poder mandar no getitem
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