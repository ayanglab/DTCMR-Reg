import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


def log_image_wandb(
    images: list[torch.Tensor],
    name: str,
):
    slices, height, width = images[0].shape
    new_image = torch.zeros((slices * height, width * len(images)))
    for i, image in enumerate(images):
        for j in range(slices):
            new_image[j * height : (j + 1) * height, i * width : (i + 1) * width] = image[j]

    wandb.log({name: [wandb.Image(new_image, caption=name)]}, commit=True)

def log_image_wandb_np(
    images: list[np.ndarray],
    name: str):
    slices, height, width = images[0].shape
    new_image = np.zeros((slices * height, width * len(images)))
    for i, image in enumerate(images):
        for j in range(slices):
            new_image[j * height : (j + 1) * height, i * width : (i + 1) * width] = image[j]

    wandb.log({name: [wandb.Image(new_image, caption=name)]}, commit=True)
    
def draw_image(
    images: list[torch.Tensor], # or list[np.ndarray]
    name: str,
    vmin: int = -400,
    vmax: int = 400):
    '''
    images should be [img1, img2, img3]
    imgi = [nt, nx, ny]
    '''
    #log_image_wandb([imgb4, warped, y_synt], 'trn/images', i)
    column = len(images)
    row = images[0].shape[0]
    
    fig, axs = plt.subplots(row, column, figsize=(column, row))
    axs = np.array(axs)
    axs = axs.reshape(row, column)

    for i in range(row):
        for j in range(column):
            axs[i, j].imshow(images[j][i], cmap='gray', vmin=vmin, vmax=vmax)
            axs[i, j].axis('off')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    image = Image.open(buf)

    wandb.log({name: [wandb.Image(image, caption=name)]}, commit=True)

def draw_image_wiz_displacement(
    images: list[torch.Tensor],# or list[np.ndarray]
    disp: torch.Tensor, 
    name: str,
    vmin: int = 0,
    vmax: int = 1,
    scale: int = 3, # larger the number, smaller the arrow
):
    '''
    This input b4warp, warped, y_synt
    for the second img2, apply the quiver plot
    images should be [img1, img2, img3]
    imgi = [nt, nx, ny]
    disp = [nt, 2, nx, ny]
    '''
    #log_image_wandb([imgb4, warped, y_synt], 'trn/images', i)
    column = len(images)
    row = images[0].shape[0]
    
    fig, axs = plt.subplots(row, column, figsize=(column, row))
    axs = np.array(axs)
    axs = axs.reshape(row, column)
    # always draw the quiver on the second deformed images
    X, Y = np.meshgrid(np.arange(images[1].shape[-2]), np.arange(images[1].shape[-1]), indexing='ij')
    for i in range(row):
        for j in range(column):
            axs[i, j].imshow(images[j][i], cmap='gray', vmin=vmin, vmax=vmax)
            if j == 1:
                # quiver
                axs[i,j].quiver(X[::5,::5], Y[::5,::5], -disp[i,1,::5,::5], disp[i,0,::5,::5], color='r', units='xy', scale=scale)
            axs[i, j].axis('off')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    image = Image.open(buf)

    wandb.log({name: [wandb.Image(image, caption=name)]}, commit=True)

def normalise_disp(disp):
    """
    Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes disp size is the same as the corresponding image.

    Args:
        disp: (numpy.ndarray or torch.Tensor, shape (nt, ndim, *size)) Displacement field

    Returns:
        disp: (normalised disp)
    """

    ndim = disp.ndim - 2

    if type(disp) is np.ndarray:
        norm_factors = 2. / np.array(disp.shape[2:])
        norm_factors = norm_factors.reshape(1, ndim, *(1,) * ndim)

    elif type(disp) is torch.Tensor:
        norm_factors = torch.tensor(2.) / torch.tensor(disp.size()[2:], dtype=disp.dtype, device=disp.device)
        norm_factors = norm_factors.view(1, ndim, *(1,)*ndim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return disp * norm_factors


def combine_flow_field(flow_field: torch.Tensor)-> torch.Tensor:
    '''
    # Generate example flow field data
    params: flow_field: (nt, 2, nx, ny) array
    output: normed_magnitude: (nt,2, nx, ny) array
    '''
    #flow_field = normalise_disp(flow_field)
    
    flow_x = flow_field[:,0,:,:]
    flow_y = flow_field[:,1,:,:]

    # Calculate the magnitude and normalize it
    magnitude = torch.sqrt(flow_x ** 2 + flow_y ** 2)
    normalized_magnitude = magnitude / torch.max(magnitude)

    return normalized_magnitude


def combine_flow_field_np(flow_field: np.ndarray)-> np.ndarray:
    '''
    # Generate example flow field data
    params: flow_field: (nt, 2, nx, ny) array
    output: normed_magnitude: (nt,2, nx, ny) array
    '''
    #flow_field = normalise_disp(flow_field)
    
    flow_x = flow_field[:,0,:,:]
    flow_y = flow_field[:,1,:,:]

    # Calculate the magnitude and normalize it
    magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)
    normalized_magnitude = magnitude / np.max(magnitude)

    return normalized_magnitude