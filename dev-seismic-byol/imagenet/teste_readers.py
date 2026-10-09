import numpy as np
from scipy.io import loadmat
from pathlib import Path
from base.ImagenetReader import *

'''aparentemente entries_train[i][2] é o wnid da imagem, entries_train[i][1] é o indice desse mesmo wnid em class_ids'''
train_entries_path = Path('/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras/entries-TRAIN.npy')
train_partition_path = Path('/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train')

val_entries_path = Path('/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras/entries-VAL.npy')
val_partition_path = Path('/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/val')

mat_path = Path('/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/meta.mat')
validation_ground_truth = Path('/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extra_files/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')

ground_truth_challenge_labels = np.loadtxt(validation_ground_truth, dtype= np.int64)

val_reader = ImagenetValReader(mat_path, val_entries_path, val_partition_path)
train_reader = ImagenetTrainReader(mat_path, train_entries_path, train_partition_path)

#TESTE DE ALINHAMENTO COM O GROUND TRUTH (VALIDAÇÃO)
def test_val_ground_truth_alignment():
    print('starting alignment test')

    assert len(val_reader) == len(ground_truth_challenge_labels)
    count_align_erros = 0
    for i in range(len(val_reader)):

        metadata = val_reader.entries[i]
        metadata_wnid = str(metadata[2])
        metadata_accused_id = val_reader.wnid_to_id[metadata_wnid]

        metadata_img_idx = int(metadata[0]) - 1
        challenge_id = ground_truth_challenge_labels[metadata_img_idx]

        if challenge_id != metadata_accused_id:
            count_align_erros += 1

    print(f'number of alignment errors: {count_align_erros}')

test_val_ground_truth_alignment()

#TESTE DA NTEGRIDADE DOS READERS
def test_entry_integrity(train_reader, val_reader):
    print('starting entry intehgrity test')
    # Validação: cada imagem deve aparecer exatamente uma vez.
    val_ids = [int(row[0]) for row in val_reader.entries]

    assert len(set(val_ids)) == len(val_ids), \
        "Existem imagens duplicadas na validação."

    assert set(val_ids) == set(range(1, 50001)), \
        "A validação não cobre exatamente os IDs 1 a 50000."

    # Treino: a combinação WNID + ID da imagem deve ser única.
    train_keys = [
        (str(row[2]), int(row[0]))
        for row in train_reader.entries
    ]

    assert len(set(train_keys)) == len(train_keys), \
        "Existem entradas duplicadas no treino."

    print("entry integrity: OK")

test_entry_integrity(train_reader, val_reader)

#TESTANDO SE TODOS OS WNIDS ESTÃO MENCIONADOS
def test_known_wnids(train_reader, val_reader):
    print('starting known wnids test')
    known_wnids = set(train_reader.wnid_to_id.keys())

    for name, reader in [
        ("treino", train_reader),
        ("validação", val_reader),
    ]:
        unknown = {
            str(row[2])
            for row in reader.entries
            if str(row[2]) not in known_wnids
        }

        assert not unknown, (
            f"WNIDs desconhecidos no {name}: {unknown}"
        )

    print("all WNIDS are known: OK")

test_known_wnids(train_reader, val_reader)


def test_image_paths(reader, n=1000):
    print('starting image path tests')
    n = min(n, len(reader))
    checked = 0

    for i in range(n):
        row = reader.entries[i]
        img_idx = int(row[0])
        wnid = str(row[2])

        if isinstance(reader, ImagenetTrainReader):
            img_path = (
                reader.partition_path
                / wnid
                / f"{wnid}_{img_idx}.JPEG"
            )
        else:
            img_path = (
                reader.partition_path
                / f"ILSVRC2012_val_{img_idx:08d}.JPEG"
            )

        assert img_path.is_file(), (
            f"Imagem inexistente na entrada {i}: {img_path}"
        )

        checked += 1

    print(f"Valid Paths: {checked}")

def test_getitem(reader, indices):
    print('starting get_item test')

    for i in indices:
        image, label = reader[i]

        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert isinstance(label, (int, np.integer))
        assert 0 <= int(label) < 1000

        row = reader.entries[i]
        wnid = str(row[2])

        expected_label = (
            reader.wnid_to_id[wnid] - 1
        )

        assert int(label) == expected_label, (
            f"Label incorreta na entrada {i}"
        )

    print(f"__getitem__: {len(indices)} samples OK")


test_getitem(train_reader, range(len(train_reader)))
test_getitem(val_reader, range(len(val_reader)))