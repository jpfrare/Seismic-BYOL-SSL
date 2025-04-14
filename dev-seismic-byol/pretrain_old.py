#!/usr/bin/env python
# coding: utf-8

# In[1]:


from functions import *


# # Variables

# ## Paths

# In[2]:


data_path = "/workspaces/shared_data/seismic/f3_segmentation/images"
annotation_path = "/workspaces/shared_data/seismic/f3_segmentation/annotations"
pretrain_logs_path = "/workspaces/Minerva-Dev/dev-seismic-byol/logs/pretrain"
pretrain_ckpt_path = "/workspaces/Minerva-Dev/dev-seismic-byol/ckpt/pretrain"


# ## Hyperparameters

# In[3]:


x = 224
input_size = (x, x)

dataset_name = "seam_ai"

learning_rate = 0.2
batch_size = 8
num_epochs = 20

single_channel = True
accelerator = "gpu"

model_name = f"pretrain_{dataset_name}_E{num_epochs}_B{batch_size}"


# ## Transforms

# In[4]:


from minerva.transforms.transform import *
from minerva.transforms.random_transform import *


# In[5]:


random_flip = RandomFlip(possible_axis=1)
random_crop = RandomCrop(crop_size=input_size)
random_rotation = RandomRotation(degrees=25, prob=1)
transpose = Transpose([2, 0, 1])
cast_to_tensor = CastTo(dtype=np.float32)

byol_transform_pipeline = TransformPipeline(
    [
        random_crop,
        random_flip,
        random_rotation,
        transpose,
        cast_to_tensor,
    ]
)


# ## Data Visualization

# In[6]:


from minerva.data.readers import TiffReader


# In[7]:


tiff_reader = TiffReader(path=data_path)
image_example = tiff_reader[0]

cropped_image = random_crop(image_example)
flipped_image = random_flip(cropped_image)
rotated_image = random_rotation(flipped_image)
final_image = byol_transform_pipeline(image_example)


# In[8]:


image_list = [cropped_image, flipped_image, rotated_image, final_image[0]]
# plot_images(image_list)


# # Dataset

# ## Readers

# In[9]:


from minerva.data.datasets import SimpleDataset


# In[10]:


train_img_reader_01 = TiffReader(path=data_path)
train_img_reader_02 = TiffReader(path=data_path)


pretrain_dataset = SimpleDataset(
    readers=[train_img_reader_01, train_img_reader_02],
    transforms=byol_transform_pipeline,
)


# # DataModule

# In[11]:


from minerva.data.data_modules import MinervaDataModule


# In[12]:


data_module = MinervaDataModule(
    train_dataset=pretrain_dataset,
    batch_size=batch_size,
    drop_last=True,
    shuffle_train=True,
    name=dataset_name,
)


# In[13]:


# Testing Data Module

data_module.setup("fit")
train_batch_x, train_batch_y = next(iter(data_module.train_dataloader()))
train_batch_x.shape, train_batch_y.shape


# # Model

# In[14]:


from minerva.models.ssl.byol import BYOL
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone


# In[15]:


backbone = DeepLabV3Backbone(num_classes=6)

model = BYOL(
    backbone=backbone,
    learning_rate=learning_rate,
)

# model


# # Pipeline

# In[16]:


from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from pathlib import Path


# ## Trainer

# In[ ]:


log_dir = Path(pretrain_logs_path) / model_name / dataset_name
ckpt_dir = Path(pretrain_ckpt_path) / model_name / dataset_name
logger = CSVLogger(log_dir, name=model_name, version=dataset_name)
ckpt_callback = ModelCheckpoint(save_top_k=1, save_last=True, dirpath=ckpt_dir)

trainer = Trainer(
    accelerator="gpu",
    logger=logger,
    callbacks=ckpt_callback,
    max_epochs=num_epochs,
)


# In[18]:


pipeline = SimpleLightningPipeline(
    model=model,
    trainer=trainer,
    log_dir=log_dir,
    save_run_status=True,
)


# In[19]:


pipeline.run(data_module, task="fit")


# In[ ]:


# In[ ]:
