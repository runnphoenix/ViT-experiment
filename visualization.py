import torch

from vit import ViT

import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
from einops import rearrange

import torchvision

def show_attention(model_path, img_path):
    # read in model and image
    model = ViT(img_size=224, patch_size=16, num_classes=2, dim=768, depth=12, n_heads=12, mlp_dim=768*4)
    model.load_state_dict(torch.load(model_path))
    image = Image.open(img_path)
    transform= torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor()
        ])
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)

    patches = model.patch_embed(img_tensor)
    patches = rearrange(patches, 'b d nw nh -> b (nw nh) d')
    transformer_input = torch.cat((model.cls_token, patches), dim=1) + model.pos_embed

    # show position Embedding
    fig = plt.figure(figsize=(8,8))
    fig.suptitle(" Visualization of position embeddings ")
    for i in range(1, model.pos_embed.shape[1]): #197
        print(model.pos_embed.shape)
        print(model.pos_embed[0, i:i+1].shape)
        print(model.pos_embed[0, 1:].shape)
        sim = torch.nn.functional.cosine_similarity(model.pos_embed[0, i:i+1], model.pos_embed[0, 1:], dim=1)
        sim = sim.reshape((14, 14)).detach().cpu().numpy()
        ax = fig.add_subplot(14, 14, i)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.imshow(sim)
    plt.show()

    # show attention matrix
    attention = model.blocks[0].attn
    qkv = attention.qkv(transformer_input).chunk(3, dim=-1)
    q,k,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=12), qkv)
    att_matrix = torch.matmul(q, k.transpose(-1, -2)) * (64 ** -0.5) #shape b=1, h=12, n=197, n
    #197*197 means for each patch, what other each patch means to it
    att_matrix = att_matrix.squeeze() #remove the 1 of shape[0], as there's only 1 image in the batch

    for i in range(12): # show each head
        plt.imshow(att_matrix[i].detach().cpu().numpy())
        plt.show()

    # show attention heatmap
    for i in range(12): # show each head
        fig = plt.figure()
        fig.suptitle("Visualization of attention")

        for j in range(1, 197): # show each patch in one head
            att_heatmap = att_matrix[i, j, 1:].reshape((14, 14)).detach().cpu().numpy()
            ax = fig.add_subplot(14, 14, j)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.imshow(att_heatmap)

        plt.show()


if __name__ =='__main__':
    show_attention('./trained_model.pth', './cat_dog/test/883.jpg')
