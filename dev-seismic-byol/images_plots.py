import numpy as np
from functions import *
from minerva.transforms.transform import *
from minerva.transforms.random_transform import *
from minerva.data.readers import TiffReader
from minerva.data.datasets import SimpleDataset
from minerva.data.data_modules import MinervaDataModule

input_size = (256, 256)
dataset_name = 'f3'
batch_size = 8
num_epochs = 20
repetition = 0
learning_rate = 0.001

# data_path = '/home/vinicius.soares/asml/datasets/tiff_data/seam_ai_N'

# data_path = '/home/vinicius.soares/asml/datasets/tiff_data/seam_ai'

# data_path = '/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation'

data_path = '/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation_N'

ckpt_path = None
log_path = None
gpus = [0]


random_flip = RandomFlip(possible_axis=[1])
random_crop = RandomCrop(crop_size=input_size)
random_rotation = RandomRotation(degrees=25, prob=0.2)
transpose_to_HWC = Transpose([1, 2, 0])
transpose_to_CHW = Transpose([2, 0, 1])    
cast_to_tensor = CastTo(dtype=np.float32)
repeat = Repeat(axis=2, n_repetitions=3)


byol_transform_pipeline = TransformPipeline([
    random_crop,
    random_flip,
    random_rotation,
    transpose_to_CHW,
    cast_to_tensor,
])

constrastive_transform = ContrastiveTransform(byol_transform_pipeline)


train_img_reader = TiffReader(path=data_path)

pretrain_dataset = SimpleDataset(
    readers=train_img_reader,
    transforms=constrastive_transform,
    return_single=True
)

data_module = MinervaDataModule(
    train_dataset=pretrain_dataset,
    batch_size=batch_size,
    drop_last=True,
    shuffle_train=True,
    name=dataset_name
)


loader = data_module.train_dataloader()
batch = next(iter(loader))

plot_images([
    batch[0][2][0],
    batch[1][2][0]],
    filename="outputs/images"
            )