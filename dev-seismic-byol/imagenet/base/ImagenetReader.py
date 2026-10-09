import numpy as np
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from scipy.io import loadmat
from .TaxonomicHandler import TaxonomicHandler
import time


''' Exemplo de shape das entries do treino e validação: y = (10095, 0, 'n01440764', ' tench Tinca tinca')
    y[0]: id da imagem, se usa para conseguir acessar a imagem na pasta de destino
    y[1]: label declarada para o desafio (não será usada)
    y[2]: wnid referente a classe contida na imagem
    y[3]: uma breve descrição da classe correspondente (não será usada)
'''

'''o arquivo matlab é um pouco complicado de se entender: em primeiro lugar, a conversão matlab -> python faz com que surjam dimensões extras fantasmas.
Seja x o arquivo do matlab carregado em python, as informações úteis estão contidas em y = x['synsets'], que é um vetor onde cada indice i (y[i][0]) se refere
a uma das classes relacionadas ao desafio (1000 classes do desafio + classes que compõem a árvore de herança do worldnet até chegar nas classes folhas (as 1000 do desafio))
obs: o id 1001 (1000 no array) se refere a classe 'Entidade' a raiz dessa árvore.
Dada a existencia das dimensões extras, cada y[i][0] contém a seguinte estrutura:
y[i][0][0][0][0] é, por redundancia, o seu id no desafio original (no caso i + 1)
y[i][0][1][0] é o seu WNID, que é seu identificador no WordNet, que é o sistema de organização de categorias que contém o ImageNet
y[i][0][2][0] é o nome da classe, para i == 1000, temos 'entity'
y[i][0][3][0] é a descrição do que é (ex: para a classe 'entity':  that which is perceived or known or inferred to have its own distinct existence (living or nonliving))
y[i][0][4][0][0] é o número de nós filhos que esse nó tem (o nó é identificado pelo seu id do desafio)
y[i][0][5][0] é um vetor de tamanho número de nós filhos com os ids desses nós
os indices 6 e 7 são meio inuteis para a nossa task então vou poupar citá-los
'''

class ImagenetReader():
    wnid_to_id: dict                                #mapeia o wnid para o id do meta.mat
    id_to_label: dict                               #mapeia o id do meta.mat para a label real (alterável)
    mat: list                                       #.mat com as informações do desafio
    entries: list                                   #entries da partição desejada (treino / validação)
    partition_path: Path                            #caminho da partição (treino / validação)

    def __init__(self, mat_path, entries_path, partition_path):
        self.mat = loadmat(mat_path)['synsets']
        self.wnid_to_id = {str(row[0][1][0]) : int(row[0][0][0][0]) for row in self.mat[:1000]}
        self.id_to_label = {i : i - 1 for i in self.wnid_to_id.values()}
        self.entries = np.load(entries_path, allow_pickle = True)
        self.partition_path = partition_path

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return NotImplementedError()
    
class ImagenetTrainReader(ImagenetReader):
        
    def __getitem__(self, idx):
        row = self.entries[idx]

        img_idx = int(row[0])
        wnid = str(row[2])

        challenge_id = self.wnid_to_id[wnid]
        label = self.id_to_label[challenge_id]

        img_path = self.partition_path / wnid / f"{wnid}_{img_idx}.JPEG"

        last_error = None

        for attempt in range(10):
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")

                return img, label

            except (UnidentifiedImageError, OSError) as e:
                last_error = e
                time.sleep(0.05 * (attempt + 1))

        raise RuntimeError(
            f"Falha ao ler {img_path} após 10 tentativas"
        ) from last_error
    
class ImagenetValReader(ImagenetReader):

    def __getitem__(self, idx):

        row = self.entries[idx]

        img_idx = int(row[0])
        wnid = str(row[2])

        challenge_id = self.wnid_to_id[wnid]
        label = self.id_to_label[challenge_id]

        img_path = Path(self.partition_path / f'ILSVRC2012_val_{img_idx:08d}.JPEG')

        last_error = None
        
        for attempt in range(10):
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")

                return img, label

            except (UnidentifiedImageError, OSError) as e:
                last_error = e
                time.sleep(0.05 * (attempt + 1))

        raise RuntimeError(
            f"Falha ao ler {img_path} após 10 tentativas"
        ) from last_error


