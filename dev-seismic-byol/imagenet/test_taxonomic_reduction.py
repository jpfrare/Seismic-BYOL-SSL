from pathlib import Path
from base.TaxonomicHandler import TaxonomicHandler
from base.ImagenetReader import ImagenetTrainReader, ImagenetValReader
import numpy as np

train_entries_path = Path('/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras/entries-TRAIN.npy')
train_partition_path = Path('/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train')

val_entries_path = Path('/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras/entries-VAL.npy')
val_partition_path = Path('/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val')

mat_path = Path('/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/meta.mat')

val_reader = ImagenetValReader(mat_path, val_entries_path, val_partition_path)
train_reader = ImagenetTrainReader(mat_path, train_entries_path, train_partition_path)

handler = TaxonomicHandler(mat_path)

def visualize_classes_per_level(handler):
    for level, classes in handler.level_to_id.items():
        example = next(iter(classes), None)
        print(f'Level: {level}, Num classes: {len(classes)}, example: {example}')

def visualize_leaf_nodes(handler):
    leaf_ids = set(range(1, 1001))
    for level, classes in handler.level_to_id.items():
        ocurrences = leaf_ids & classes


        print(f'Leafs in level {level}: {len(ocurrences)}')

def validate_levels(handler):
    original_ids = set(range(1, 1001))

    # Todos os IDs originais foram alcançados?
    missing = original_ids - handler.id_to_level.keys()
    print(f"Original classes not reached: {len(missing)}")

    # Cada nó aparece em exatamente um nível?
    level_ids = set()
    duplicates = set()

    for ids in handler.level_to_id.values():
        duplicates.update(level_ids & ids)
        level_ids.update(ids)

    print(f"Nodes in multiple levels: {len(duplicates)}")
    print(f"Total distinct nodes by level: {len(level_ids)}")
    print(f"Total nodes in id_to_level: {len(handler.id_to_level)}")

    assert not missing
    assert not duplicates
    assert level_ids == set(handler.id_to_level)

visualize_classes_per_level(handler)
visualize_leaf_nodes(handler)
validate_levels(handler)
