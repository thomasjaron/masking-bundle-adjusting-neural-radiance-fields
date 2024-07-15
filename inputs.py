"""Small library to deal with our input data"""

import torch
import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps
import imageio
import torchvision.transforms.functional as torchvision_F
from easydict import EasyDict as edict

import cv2
import numpy as np

def load_images(fps, opt, mode='RGB', invert_gray=False):
    """Loads a set of images into a tensor from a list of file pointers.
    Given the model options, also scales these images and saves them to
    either cpu or gpu."""
    if not fps:
        return None
    if not isinstance(fps, list):
        raise TypeError("Function requires list of input filepaths!")
    loaded_images = []
    for fp in fps:
        im = PIL.Image.open(fp).convert(mode)
        if opt.use_cropped_images:
            im.thumbnail((opt.patch_W, opt.patch_H), PIL.Image.Resampling.LANCZOS)
        im = torchvision_F.to_tensor(im).to(opt.device)
        if mode == 'L' and invert_gray:
            im = (im < 0.5).float()
        loaded_images.append(im)
    return torch.stack(loaded_images)

def save_images(images, suffix, mode='RGB'):
    """Save image(s) from a tensor to a .png image into the root folder.
    This function is useful for debugging purposes."""
    for i, im in enumerate(images):
        gr = torchvision_F.to_pil_image(im, mode=mode)
        # frame = (gr * 255).cpu().byte().permute(1, 2, 0).numpy()
        imageio.imsave(f"{i}-{suffix}.png", gr)

def load_single_image(fp, device, mode='RGB'):
    """Load a single image with PIL and convert it to a tensor."""
    if not fp or not device:
        raise ValueError("Function requires file pointer as string and device to store tensor to.")
    im = PIL.Image.open(fp).convert(mode)
    return torchvision_F.to_tensor(im).to(device)

def compute_histograms(images, device):
    """Compute histograms from a tensor holding images."""
    # TODO
    # Images = tensor of size [B, 3, H, W]
    # device = string
    return 0

def compute_edges(images_tensor, device):
    """Compute edge image tensors from a grayscale image tensor containing 1 or more images."""
    # Images = tensor of size [B, 3, H, W]
    # device = string
    processed_images = []
    for image in images_tensor:
        # image preparation
        if image.is_cuda:
            image = image.detach().cpu()
        i = image.numpy()
        i = np.transpose(i, (1, 2, 0))
        # compute derivatives and edge image
        sobel_x = cv2.Sobel(i, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(i, cv2.CV_64F, 0, 1, ksize=3)
        i = np.sqrt(sobel_x**2 + sobel_y**2)
        # blur result to make loss calculation and gradient descent easier
        i = cv2.GaussianBlur(i, (5, 5), 0)
        i = torchvision_F.to_tensor(i).to(device)
        processed_images.append(i)
    return torch.stack(processed_images)

def erode_images(images_tensor, device, kernel=(5, 5)):
    """Compute eroded grayscale image tensor containing 1 or more images."""
    # Images = tensor of size [B, x, H, W]
    # device = string
    processed_images = []
    for image in images_tensor:
        # image preparation
        if image.is_cuda:
            image = image.detach().cpu()
        i = image.numpy()
        i = np.transpose(i, (1, 2, 0))
        i = cv2.erode(i, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
        i = torchvision_F.to_tensor(i).to(device)
        processed_images.append(i)
    return torch.stack(processed_images)


def prepare_images(opt, fps_images=None, fps_masks=None, fp_gt=None, edges=True):
    """Load distorted and occluded images used for reconstruction.
    This function assumes a 
    """
    inputs = edict()
    # load groundtruth
    inputs.gt = load_single_image(fp_gt, opt.device)
    # load images from dataset
    inputs.rgb = load_images(fps_images, opt)
    # Invert loaded masks (SIDAR Dataset sets occlusions to 1)
    inputs.masks = load_images(fps_masks, opt, mode='L', invert_gray=True)
    inputs.masks_eroded = erode_images(inputs.masks, opt.device, kernel=(5,5)) if inputs.masks else None
    ##### perform image processing and save the results
    # save grayscale version of images
    inputs.gray = load_images(fps_images, opt, mode='L')
    # generate edge images
    inputs.edges = compute_edges(inputs.gray, opt.device) if edges else None
    # TODO:
    # inputs.histograms = compute_histograms(inputs.rgb, opt.device)
    # inputs.histograms_normalized = compute_histograms(inputs.rgb, opt.device)

    return inputs
