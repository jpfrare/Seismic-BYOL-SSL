import numpy as np
data = np.load("/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/extras/entries-VAL.npy", allow_pickle=True)
print(f"Min class_index: {data['class_index'].min()}")
print(f"Max class_index: {data['class_index'].max()}")