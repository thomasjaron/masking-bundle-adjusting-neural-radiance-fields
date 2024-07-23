"""Module for utility function for visualization"""
import torch
import torchvision
import matplotlib.pyplot as plt

# pylint: disable=invalid-name


@torch.no_grad()
def tb_image(opt, tb, step, group, name, images, num_vis=None, from_range=(0, 1), cmap="gray"):
    """Add given image(s) to tensorboard"""
    images = preprocess_vis_image(images, from_range=from_range, cmap=cmap)
    num_H, num_W = num_vis or opt.tb.num_images
    images = images[:num_H*num_W]
    image_grid = torchvision.utils.make_grid(
        images[:, :3], nrow=num_W, pad_value=1.)
    if images.shape[1] == 4:
        mask_grid = torchvision.utils.make_grid(
            images[:, 3:], nrow=num_W, pad_value=1.)[:1]
        image_grid = torch.cat([image_grid, mask_grid], dim=0)
    tag = f"{group}/{name}"
    tb.add_image(tag, image_grid, step)


def preprocess_vis_image(images, from_range=(0, 1), cmap="gray"):
    """Preprocess image before adding to tensorboard"""
    min_val, max_val = from_range
    images = (images-min_val)/(max_val-min_val)
    images = images.clamp(min=0, max=1).cpu()
    if images.shape[1] == 1:
        images = get_heatmap(images[:, 0].cpu(), cmap=cmap)
    return images


def get_heatmap(gray, cmap):  # [N,H,W]
    """Get heatmap for 1D Images"""
    color = plt.get_cmap(cmap)(gray.numpy())
    color = torch.from_numpy(color[..., :3]).permute(
        0, 3, 1, 2).float()  # [N,3,H,W]
    return color


def color_border(images, colors, width=3, depth=3):
    """Add padding colored borders to images before adding them to tensor board"""
    images_pad = []
    for i, image in enumerate(images):
        if depth == 1:
            image_pad = torch.ones(
                1, image.shape[1]+width*2, image.shape[2]+width*2)*(127.0/255.0)
        if depth == 3:
            image_pad = torch.ones(
                3, image.shape[1]+width*2, image.shape[2]+width*2)*(colors[i, :, None, None]/255.0)
        image_pad[:, width:-width, width:-width] = image
        images_pad.append(image_pad)
    images_pad = torch.stack(images_pad, dim=0)
    return images_pad # [B, depth, H, W]
