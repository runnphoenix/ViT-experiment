import torch

from vit import ViT

import matplotlib.pyplot as plt

def show_attention(model_path, img_path):
    # read in model and image
    model = ViT( ** parameters)
    model.load_state_dict(torch.load(model_path))
    image = Image.read(img_path)

    model(image)
    
    att_matrix = model.blocks[0].attn
    print("attention matrix:{}".format(att_matrix.shape))
    plt.imshow(att_matrix[3].detach().cpu().numpy())

    fig = plt.figure()
    fig.suptitle("Visualization of attenion")
    fig.add_axes()
    image = np.asarray(image)
    ax = fig.add_subplot(2,4,1)
    ax.imshow(image)

    for i in range(7):
        att_heatmap = att_matrix[i, 100, 1:].reshape((14,14)).detach().cpu().numpy()
        ax = fig.add_subplot(2,4,i+2)
        ax.imshow(att_heatmap)

