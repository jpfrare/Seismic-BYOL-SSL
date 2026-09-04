from torchvision.datasets import ImageFolder
import numpy as np
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from scipy.io import loadmat
from .utils import reduce_taxonomic_diversity
import time

#por alguma razão mistica, tanto o dataset de teste como de treino sao arrays estruturados do numpy
#pra cada linha do array do treino:
#linha[0] é o id do arquivo ex: 10026
#linha[1] é a label dele
#linha[2] é o prefixo/subpasta ex: n01440764 (wnid)
#você consegue fazer o caminho da imagem juntando essas informações e a label obviamente está na linha[1], ai você forma uma sample
class ImagenetReader:
    def __init__(self, root, entries_path):
        self.root = Path(root)
        # Carrega o array estruturado 
        self.data = np.load(entries_path, allow_pickle=True)
        print(f' tamanho das entries do treino: {len(self.data)}')
        
        # Mapeia as labels (coluna index 1) para o StratifiedSubset
        # Fazemos isso no init para o subset não precisar iterar depois
        self.targets = [int(row[1]) for row in self.data]
        self.wind_to_coarse = None

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data[idx]

        img_idx = row[0]
        wnid = row[2]

        label = (
            self.wind_to_coarse[wnid]
            if self.wind_to_coarse is not None
            else int(row[1])
        )

        img_path = self.root / wnid / f"{wnid}_{img_idx}.JPEG"

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
            f"Falha ao ler {img_path} após 3 tentativas"
        ) from last_error
    
    def to_coarse_classes(self, top_down: bool, level: int, mat_path: str):
        unique_winds = sorted(list(set(row[2] for row in self.data)))

        self.wind_to_coarse, num_classes = reduce_taxonomic_diversity(unique_winds, top_down, level, mat_path)
        self.targets = [self.wind_to_coarse[row[2]] for row in self.data]
    
        return num_classes

'''o arquivo matlab é um pouco complicado de se entender: em primeiro lugar, a conversão matlab -> python faz com que surjam dimensões extras fantasmas.
Seja x o arquivo do matlab carregado em python, as informações úteis estão contidas em y = x['sysnets'], que é um vetor onde cada indice i (y[i][0]) se refere
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

class ImagenetValReader():
    def __init__(self, root, gt_path, mat_path):
        self.root = Path(root)
        
        meta = loadmat(mat_path)['synsets']
        id_to_wnid = {int(m[0][0][0][0]): str(m[0][1][0]) for m in meta[:1000]} #ids arbitrários (1 a 1000) para o wnid respectivo
        
        self.all_wnids = sorted([str(m[0][1][0]) for m in meta[:1000]]) #ordena os winds
        self.wnid_to_label = {wnid: i for i, wnid in enumerate(self.all_wnids)} #baseado na ordenação, atribui um label (indice posicional)
        id_to_label = {comp_id: self.wnid_to_label[wnid] for comp_id, wnid in id_to_wnid.items()} #faz a ponte entre o id arbitrário e o label atribuido
        
        #lê o ground truth, é o arquivo que faz a ponte (linha x contém o id arbitrário, mas x também é o valor numérico
        #que faz parte do caminho da imagem correspondente (ILSVRC2012_val_x.JPEG) -> entao se faz um vetor onde na posição x temos o indice arbitrário
        with open(gt_path, 'r') as f:
            raw_ids = [int(line.strip()) for line in f.readlines()]
            
        self.targets = [id_to_label[id_bruto] for id_bruto in raw_ids] #faz a conversão do id arbitrário para a label dada

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Localiza a imagem (ILSVRC2012_val_00000001.JPEG)
        img_name = f"ILSVRC2012_val_{idx+1:08d}.JPEG"
        img_path = self.root / img_name
        
        # Retorna a imagem bruta (PIL) e a label traduzida
        label = self.targets[idx]

        for attempt in range(10):
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                return img, label
            except(UnidentifiedImageError, OSError) as e:
                last_error = e
                time.sleep(0.05 * (attempt + 1))
        raise RuntimeError('Falha de Leitura na Validação') from e
    
    def to_coarse_classes(self, top_down: bool, level: int, mat_path: str):
        wnid_to_coarse, num_classes = reduce_taxonomic_diversity(self.all_wnids, top_down, level, mat_path)
        
        old_label_to_new_label = {self.wnid_to_label[wnid] : wnid_to_coarse[wnid] for wnid in self.all_wnids}
        self.targets = [old_label_to_new_label[label] for label in self.targets]

        return num_classes


