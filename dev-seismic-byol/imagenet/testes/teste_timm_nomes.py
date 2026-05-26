from torchvision.models.resnet import resnet50
import timm

timm_state_dict = timm.create_model('resnet50', pretrained=False, output_stride=16, num_classes= 0).state_dict()
torch_state_dict = resnet50(replace_stride_with_dilation=[False, True, True]).state_dict()

print(timm_state_dict.keys())
print(torch_state_dict.keys())