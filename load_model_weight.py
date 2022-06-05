import torch
from timm import create_model as creat

model = creat("vit_base_patch16_224", pretrained=False)

model.load_state_dict(torch.load("./jx_vit_base_p16_224-80ecf9dd.pth"))

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
