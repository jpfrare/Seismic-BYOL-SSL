import random
from collections import defaultdict
from typing import Any, Tuple, Union

import torch
from torch.utils.data import Dataset, Subset

# Importando a base da Minerva
from minerva.data.datasets.base import SimpleDataset

class ImagenetDataset(SimpleDataset):
    def __init__(self, ImagenetReader, transform):
        super().__init__(readers= ImagenetReader, transforms= transform, return_single= False)
    
    def __getitem__(self, idx: int) -> Union[Any, Tuple[Any, ...]]:
        reader = self.readers[0]
        transform = self.transforms[0]

        img, label = reader[idx]
        sample = (transform(img), label)

        return sample

class StratifiedSubset(Subset):
    def __init__(self, dataset, per_class, seed, num_classes):
        labels = dataset.readers[0].targets  #vetor onde a labels[i] retorna a classe da imagem i

        rng = random.Random(seed)

        class_to_indices = {}               #dicionário cuja chave é uma classe e seu elemento é uma lista das posições em labels que pertencem a essa classe
        for idx in range(len(labels)):
            if labels[idx] in class_to_indices:
                class_to_indices[labels[idx]].append(idx)
            else:
                class_to_indices[labels[idx]] = [idx]

        classes = sorted(list(class_to_indices.keys()))
        rng.shuffle(classes)                #embaralha as classes

        if num_classes == 1000:
            selected_classes = classes
            self.class_mapping = None
        else:
            selected_classes = classes[:num_classes]    #seleciona as num_classes primeiras (como está desordenado, é uma escolha cumulativa e aleatória)
            self.class_mapping = {selected_classes[i] : i for i in range(num_classes)} #remapeia as classes pra poder ter labels entre 0 e num_classes
        
        
        
        selected = []

        # amostra balanceada
        for c in selected_classes:
            idxs = list(class_to_indices[c])
            rng.shuffle(idxs)
            k = min(per_class, len(idxs))
            selected.extend(idxs[:k])

        # embaralha ordem final
        rng.shuffle(selected)

        super().__init__(dataset, selected)
    
    def __getitem__(self, idx : int):
        img, label = super().__getitem__(idx)

        if self.class_mapping is not None:
            return img, self.class_mapping[label]
        return img, label