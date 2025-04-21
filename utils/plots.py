import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def generate_images(fields):
    inp, tar, gen = [x.detach().float().cpu().numpy() for x in fields]
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    plt.title('2m temperature')
    ax[0].imshow(tar[0,2,:,:], cmap="turbo")
    ax[0].set_title("ERA5 ground truth")
    ax[1].imshow(gen[0,2,:,:], cmap="turbo")
    ax[1].set_title("ViT prediction (patch=8)")
    fig.tight_layout()
    fig.savefig("output_image.png", dpi=300)
    return fig
