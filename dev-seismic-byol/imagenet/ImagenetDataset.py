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
    def __init__(self, dataset, per_class, seed=42):
        labels = dataset.readers[0].targets  # assume seu reader

        rng = random.Random(seed)

        class_to_indices = {}
        for idx in range(len(labels)):
            if labels[idx] in class_to_indices:
                class_to_indices[labels[idx]].append(idx)
            else:
                class_to_indices[labels[idx]] = [idx]

        classes = list(class_to_indices.keys())
        n_classes = len(classes)

        selected = []

        # amostra balanceada
        for c in classes:
            idxs = class_to_indices[c]
            k = min(per_class, len(idxs))
            selected.extend(rng.sample(idxs, k))

        # embaralha ordem final
        rng.shuffle(selected)

        super().__init__(dataset, selected)