
from PIL import Image

path = "/petrobr/parceirosbr/spfm/datasets/ImageNet_2012/train/n04090263/n04090263_5656.JPEG"

with Image.open(path) as img:
    print("format:", img.format)
    print("size:", img.size)
    print("mode:", img.mode)
    img.verify()

print("JPEG válido")