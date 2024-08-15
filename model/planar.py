"""Contains Necessary classes for planar BARF"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_F
import torchvision.transforms.functional as torchvision_F
from torch.utils import tensorboard
import tqdm
from easydict import EasyDict as edict
import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps
import imageio
import visdom
import cv2
import inputs

import util
import util_vis
from util import log
from warp import Warp

import matplotlib.pyplot as plt

import scipy.io
import kornia

# ============================ main engine for training and evaluation ============================

class Model(torch.nn.Module):
    """DL Model for Planar BARF"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.dataset = opt.dataset
        os.makedirs(opt.output_path, exist_ok=True)
        self.warp = Warp(opt)
        # container for all images and image processings
        self.images = None
        # build networks
        self.graph = None
        # setup optimizer
        self.optim = None
        self.sched = None

        self.tb = None

        self.box_colors = None
        self.vis_path = None
        self.video_fname = None
        self.timer = None
        self.warp_pert = None
        self.ep = self.it = self.vis_it = 0
        self.gt_hom = None

    def load_dataset(self):
        """Load all images and other inputs."""
        log.info("loading dataset...")
        image_paths = [
            f'data/planar/{self.dataset}/{i}.png' for i in range(self.batch_size)
        ]
        mask_paths = [
            f'data/planar/{self.dataset}/{i}-m.png' for i in range(self.batch_size)
        ]
        self.images = inputs.prepare_images(
            self.opt,
            fps_images=image_paths,
            fps_masks=mask_paths if (self.opt.use_masks and not self.opt.use_implicit_mask) else None,
            fp_gt=f'data/planar/{self.dataset}/gt.png',
            edges = True if self.opt.use_edges else None
            )
        self.gt_hom = torch.stack([
            torch.tensor(np.loadtxt('data/planar/cat_batch2/H_0_1.mat')),
            torch.tensor(np.loadtxt('data/planar/cat_batch2/H_0_2.mat')),
            torch.tensor(np.loadtxt('data/planar/cat_batch2/H_0_3.mat')),
            torch.tensor(np.loadtxt('data/planar/cat_batch2/H_0_4.mat')),
            torch.tensor(np.loadtxt('data/planar/cat_batch2/H_0_5.mat'))
        ])

    def build_networks(self):
        """Builds Network"""
        log.info("building networks...")
        self.graph = Graph(self.opt).to(self.opt.device)

    def setup_optimizer(self):
        """Set up optimizer and add networks to it"""
        log.info("setting up optimizers...")
        optim_list = [
            dict(params=self.graph.neural_image.parameters(), lr=self.opt.optim.lr),
            dict(params=self.graph.warp_param.parameters(), lr=self.opt.optim.lr_warp),
        ]
        if self.opt.use_implicit_mask and self.opt.build_single_masks:
            optim_list.extend([dict(params=mask.parameters(), lr=self.opt.optim.lr_mask) for (i, mask) in self.graph.implicit_masks.items()])
        elif self.opt.use_implicit_mask:
            optim_list.append(dict(params=self.graph.implicit_mask.parameters(), lr=self.opt.optim.lr_mask))

        optimizer = getattr(torch.optim, self.opt.optim.algo)
        self.optim = optimizer(optim_list)
        # set up scheduler
        if self.opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler, self.opt.optim.sched.type)
            kwargs = { k:v for k, v in self.opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim, **kwargs)

    def setup_visualizer(self):
        """Setup for visualization"""
        log.info("setting up visualizers...")
        # activate tensorboard
        if self.opt.tb:
            self.tb = torch.utils.tensorboard.SummaryWriter(log_dir=self.opt.output_path, flush_secs=10)

        # Prepare homography visualization
        box_colors = [
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#FFFF00",  # Yellow
            "#00FFFF",  # Cyan
            "#FF00FF",  # Magenta
            "#800000",  # Maroon
            "#808000",  # Olive
            "#008080",  # Teal
            "#800080",  # Purple
            "#808080"   # Gray
        ]
        box_colors = box_colors[:self.batch_size]
        box_colors = list(map(util.colorcode_to_number, box_colors))
        self.box_colors = np.array(box_colors).astype(int)

        # create visualization and output directories
        self.vis_path = f"{self.opt.output_path}/vis"
        os.makedirs(self.vis_path, exist_ok=True)
        self.video_fname = f"{self.opt.output_path}/vis.mp4"

    def train(self, mode=True):
        """Train Model"""
        # Training Preparations
        log.title("TRAINING START")
        self.timer = edict(start=time.time(), it_mean=None)

        # set Graph to training mode
        self.graph.train()
        # add training images with masks to var
        var = edict(idx=torch.arange(self.batch_size))
        var.images = self.images

        # Training Loop
        var = util.move_to_device(var, self.opt.device)
        loader = tqdm.trange(self.opt.max_iter, desc="Training", leave=False)
        # Manually start first iteration for visualization
        var = self.graph.forward(var)
        self.visualize(var, step=0)
        for _ in loader:
            # train iteration
            _ = self.train_iteration(var, loader)
            if self.opt.warp.fix_first:
                self.graph.warp_param.weight.data[0] = 0


        # after training
        # generate video
        os.system(
            f"ffmpeg -y -framerate 30 -i {self.vis_path}/%d.png -pix_fmt yuv420p {self.video_fname}"
        )
        # clear tensorboard and visualization
        if self.opt.tb:
            self.tb.flush()
            self.tb.close()
        log.title("TRAINING DONE")

    def summarize_loss(self, loss):
        """Summarize loss by applying the predefined weights from the options."""
        loss_all = 0.
        assert "all" not in loss
        # weigh losses
        for key in loss:
            assert key in self.opt.loss_weight
            assert loss[key].shape==()
            if self.opt.loss_weight[key] is not None:
                assert not torch.isinf(loss[key]), f"loss {key} is Inf"
                assert not torch.isnan(loss[key]), f"loss {key} is NaN"
                loss_all += 10**float(self.opt.loss_weight[key]) * loss[key]
        loss.update(all=loss_all)
        return loss

    def train_iteration(self, var, loader):
        """Train one iteration"""
        # before train iteration
        self.timer.it_start = time.time()
        # train iteration
        self.optim.zero_grad() # reset gradients
        var = self.graph.forward(var,mode="train")
        loss = self.graph.compute_loss(var,mode="train")
        loss = self.summarize_loss(loss)
        loss.all.backward()
        self.optim.step()
        # after train iteration
        if (self.it + 1) % self.opt.freq.scalar == 0:
            if self.tb:
                self.log_scalars(loss, step=self.it + 1, split="train")
        if (self.it + 1 ) % self.opt.freq.vis == 0:
            self.visualize(var, step=self.it + 1, split="train")
        self.it += 1
        loader.set_postfix(it=self.it, loss=f"{loss.all:.3f}")
        self.timer.it_end = time.time()
        util.update_timer(self.opt, self.timer, self.ep, len(loader))
        self.graph.neural_image.progress.data.fill_(self.it / self.opt.max_iter)
        return loss

    @torch.no_grad()
    def predict_entire_image(self):
        """Retrieve the full size image from the implicit neural image function"""
        xy_grid = self.warp.get_normalized_pixel_grid()[:1]
        rgb = self.graph.neural_image.forward(xy_grid) # [B, HW, 3]
        image = rgb.view(self.opt.H, self.opt.W, 3).detach().cpu().permute(2, 0, 1)
        return image

    @torch.no_grad()
    def log_scalars(self, loss, metric=None, step=0, split="train"):
        """Log scalar values into the tensorboard instance"""
        for key,value in loss.items():
            if key=="all":
                continue
            if self.opt.loss_weight[key] is not None:
                self.tb.add_scalar(f"{split}/loss_{key}",value,step)
        if metric is not None:
            for key,value in metric.items():
                self.tb.add_scalar(f"{split}/{key}",value,step)
        # compute PSNR
        print("warp_param weight shape:", self.graph.warp_param.weight.shape)
        print("gt_hom shape:", self.gt_hom.shape)
        
        # Convert warp_param.weight to homography matrix form
        warp_matrices = torch.zeros((5, 3, 3), dtype=self.graph.warp_param.weight.dtype)
        warp_matrices[:, 0, 0] = self.graph.warp_param.weight[:, 0]
        warp_matrices[:, 0, 1] = self.graph.warp_param.weight[:, 1]
        warp_matrices[:, 0, 2] = self.graph.warp_param.weight[:, 2]
        warp_matrices[:, 1, 0] = self.graph.warp_param.weight[:, 3]
        warp_matrices[:, 1, 1] = self.graph.warp_param.weight[:, 4]
        warp_matrices[:, 1, 2] = self.graph.warp_param.weight[:, 5]
        warp_matrices[:, 2, 0] = self.graph.warp_param.weight[:, 6]
        warp_matrices[:, 2, 1] = self.graph.warp_param.weight[:, 7]
        warp_matrices[:, 2, 2] = 1.0  # Homogeneous coordinate
        
        # Normalize homographies from pixel to normalized coordinates [-1, +1]
        norm_warp_matrices = kornia.geometry.conversions.normalize_homography(warp_matrices, (360, 480),(360, 480))
        norm_gt_hom = kornia.geometry.conversions.normalize_homography(self.gt_hom, (360, 480),(360, 480))
        print(f"warp_matrices {warp_matrices}")
        print(f"gt_hom {self.gt_hom}")
        #warp_error = (self.graph.warp_param.weight-self.gt_hom).norm(dim=-1).mean()
        #warp_error = (warp_matrices-self.gt_hom).norm(dim=-1).mean()
        #norm_warp_matrices = kornia.geometry.conversions.normalize_homography(warp_matrices, [-1,+1], [-1,+1])
        #norm_gt_hom = kornia.geometry.conversions.normalize_homography(self.gt_hom, [0,479], [0,359])
        print(f"norm_warp_matrices {norm_warp_matrices}")
        print(f"norm_gt_hom {norm_gt_hom}")
        warp_error = ((norm_warp_matrices / torch.det(norm_warp_matrices).abs().pow(1/3).unsqueeze(-1).unsqueeze(-1)) -
                    (norm_gt_hom / torch.det(norm_gt_hom).abs().pow(1/3).unsqueeze(-1).unsqueeze(-1))).norm(dim=(1, 2)).mean()
        #warp_error = (norm_warp_matrices-norm_gt_hom).norm(dim=-1).mean()
        print(f"warp_error {warp_error}")
        self.tb.add_scalar(f"{split}/Homography_Error", warp_error, step)
        psnr = -10 * loss.render.log10()
        self.tb.add_scalar(f"{split}/PSNR", psnr, step)

    @torch.no_grad()
    def visualize(self, var, step=0, split="train"):
        """Perform preparations for the training visualization after completion of the rendering and load
        current prediction images into tensorboard"""
        # dump frames for writing to video
        frame = self.predict_entire_image()
        imageio.imsave(f"{self.vis_path}/{self.vis_it}.png", (frame * 255).byte().permute(1, 2, 0).numpy())

        self.vis_it += 1
        # visualize in Tensorboard
        if self.opt.tb:
            colors = self.box_colors
            util_vis.tb_image(
                self.opt, self.tb, self.it+1, "train", "input_images", util_vis.color_border(var.images.rgb, colors) # pylint: disable=line-too-long
                )
            util_vis.tb_image(
                self.opt, self.tb, self.it+1, "train", "predicted_image", frame[None]
                )
            # Print out Masks
            if self.opt.use_implicit_mask:
                mask_formed = var.mask_prediction.view(self.batch_size, int(self.opt.patch_H), int(self.opt.patch_W), 1).permute(0, 3, 1, 2).cpu()
                util_vis.tb_image(
                    self.opt, self.tb, self.it+1, "train", "implicit_masks", util_vis.color_border((mask_formed), colors, width=1, depth=1) # pylint: disable=line-too-long
                    )
            # Print out Predictions Edges
            if self.opt.use_edges:
                edge_formed = var.edge_prediction.view(self.batch_size, int(self.opt.patch_H), int(self.opt.patch_W), 3).permute(0, 3, 1, 2).cpu()
                util_vis.tb_image(
                    self.opt, self.tb, self.it+1, "train", "predicted_edges", edge_formed
                    )

# ============================ Computation Graph for forward/backprop ============================

class Graph(torch.nn.Module):
    """Graph for planar BARF
    This class represents the homographies from input images / patches to correct image positions.
    This class also pytorch-internally holds the computation graph necessary to perform backward
    propagation in the network in regards to the loss function value.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.neural_image = NeuralImageFunction(opt)
        self.warp = Warp(opt)
        # represents the homographies for each input image
        self.warp_param = torch.nn.Embedding(self.batch_size, opt.warp.dof).to(opt.device)
        torch.nn.init.zeros_(self.warp_param.weight)
        # Utility variables
        self.h = self.opt.patch_H if self.opt.use_cropped_images else self.opt.H
        self.w = self.opt.patch_W if self.opt.use_cropped_images else self.opt.W
        # Iterations
        self.max_iter = opt.max_iter
        self.it = 0

        if self.opt.use_implicit_mask:
            self.embedding_uv = PosEmbedding(10-1, 10)
            if self.opt.build_single_masks:
                self.implicit_masks = {}
                for i in range(self.batch_size):
                    self.implicit_masks[f'{i}'] = ImplicitMask()
            else:
                self.implicit_mask = ImplicitMask()
            self.embedding_view = torch.nn.Embedding(self.opt.N_vocab, 128)

    def forward(self, var, mode=None): # pylint: disable=unused-argument
        """Get image and mask predictions given the current homographies"""
        xy_grid = self.warp.get_normalized_pixel_grid(crop=self.opt.use_cropped_images)
        ############ Neural Image Prediction ###########################
        xy_grid_warped = self.warp.warp_grid(xy_grid, self.warp_param.weight)
        var.rgb_prediction = self.neural_image.forward(xy_grid_warped) # [B, HW, 3]
        var.rgb_prediction_map = var.rgb_prediction.view(self.batch_size, int(self.h), int(self.w), 3).permute(0, 3, 1, 2) # [B, 3, H, W]
        var.edge_prediction = inputs.compute_edges(var.rgb_prediction_map, self.opt.device) # [B, 3, H, W]
        ############ Implicit Mask Generation ##########################
        masks = []
        if self.opt.use_implicit_mask:
            for i, im in enumerate(var.images.rgb):
                # Prepare the input for the implicit mask network(s)
                flattened_image = im.long().view(3, -1).permute(1, 0)
                uv_embedded = self.embedding_uv(xy_grid[0])
                view_embedded = self.embedding_view(flattened_image).view(180, 240, 3, -1)
                embedded_image_flat = view_embedded.view(-1, 3 * 128)
                if self.opt.build_single_masks:
                    p = self.implicit_masks[f'{i}'](torch.cat((embedded_image_flat, uv_embedded), dim=-1).cpu())
                else:
                    p = self.implicit_mask(torch.cat((embedded_image_flat, uv_embedded), dim=-1))
                masks.append(p)
            var.mask_prediction = torch.stack(masks).to(self.opt.device)
            var.mask_prediction_map = var.mask_prediction.view(self.batch_size, int(self.h), int(self.w), 1).permute(0, 3, 1, 2) # [B, 1, H, W]
        return var

    def compute_loss(self, var, mode=None): # pylint: disable=unused-argument
        """Compute Losses"""
        loss = edict()
        # Influence factor for edge alignment and rgb alignment in loss
        alpha = self.opt.alpha_initial + (self.opt.alpha_final - self.opt.alpha_initial) * (self.it / self.max_iter) if self.opt.use_edges else 0

        if self.opt.loss_weight.render is not None:
            rgb_loss = self.mse_loss(
                var.rgb_prediction_map,
                var.images.rgb,
                var.mask_prediction_map if self.opt.use_implicit_mask else var.images.masks)
            edge_loss = self.mse_loss(
                var.edge_prediction,
                var.images.edges,
                var.mask_prediction_map if self.opt.use_implicit_mask else var.images.masks_eroded) if self.opt.use_edges else torch.tensor(0)
            mask_loss = ((1 - var.mask_prediction_map.contiguous())**2).mean() if self.opt.use_implicit_mask else torch.tensor(0)
            loss.render = \
                (1 - alpha) * rgb_loss + \
                0.5 * mask_loss + \
                (alpha) * edge_loss

            loss.rgb = rgb_loss
            loss.mask = mask_loss
            loss.edge = edge_loss
        self.it += 1
        return loss

    def mse_loss(self, pred, labels, masks=None):
        """Perform MSE on prediction and groundtruth images and use masks if available."""
        if masks is None:
            loss = (pred.contiguous() - labels) ** 2
            loss = loss.mean()
        else:
            masked_diff = (pred.contiguous() - labels) * masks
            masked_loss = masked_diff ** 2
            loss = masked_loss.sum() / masks.sum()  # Only average over unmasked elements
        return loss

# ============================ Neural Image Function ============================

class NeuralImageFunction(torch.nn.Module):
    """Neural Image Function for planar BARF
    This class represents the image function, yielding us an image for given 2d
    coordinates. The image function is based on a neural network in barfs case.
    Positional encoding for the 2d coordinate grid can be performed.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.define_network()
        self.progress = torch.nn.Parameter(
            torch.tensor(0.)
            ) # use Parameter so it could be checkpointed

    def define_network(self):
        """Define Network"""
        input_2d_dim = 2 + 4 * self.opt.arch.posenc.L_2D if self.opt.arch.posenc else 2
        # point-wise RGB prediction
        self.mlp = torch.nn.ModuleList()
        ll = util.get_layer_dims(self.opt.arch.layers)
        for li, (k_in, k_out) in enumerate(ll):
            if li == 0:
                k_in = input_2d_dim
            if li in self.opt.arch.skip:
                k_in += input_2d_dim
            linear = torch.nn.Linear(k_in, k_out)
            if self.opt.barf_c2f and li == 0:
                # rescale first layer init (distribution was for pos.enc. but only xy is first used)
                scale = np.sqrt(input_2d_dim/2.)
                linear.weight.data *= scale
                linear.bias.data *= scale
            self.mlp.append(linear)

    def forward(self, coord_2d): # [B, ..., 3]
        """Generate RGB image from input grid."""
        # perform positional encoding
        if self.opt.arch.posenc:
            points_enc = self.positional_encoding(coord_2d)
            points_enc = torch.cat([coord_2d, points_enc], dim=-1) # [B, ..., 6L+3]
        else:
            points_enc = coord_2d
        feat = points_enc
        # extract implicit features
        for li, layer in enumerate(self.mlp):
            if li in self.opt.arch.skip:
                feat = torch.cat([feat, points_enc], dim=-1)
            # apply features to current layer
            feat = layer(feat)
            if li != len(self.mlp) - 1:
                # apply ReLU to every inner output
                feat = torch_F.relu(feat)
        # apply sigmoid to the final output
        rgb = feat.sigmoid_() # [B, ..., 3]
        return rgb

    def positional_encoding(self, coord_2d): # [B, ..., N]
        """Perform positional encoding on input 2d array"""
        ll = self.opt.arch.posenc.L_2D
        shape = coord_2d.shape
        # create sin/cos array of fitting size for the input
        freq = 2**torch.arange(ll, dtype=torch.float32, device=self.opt.device) * np.pi # [ll]
        spectrum = coord_2d[..., None]*freq # [B, ..., N, L]
        sin, cos = spectrum.sin(), spectrum.cos() # [B, ..., N, L]
        input_enc = torch.stack([sin, cos], dim=-2) # [B, ..., N, 2, L]
        input_enc = input_enc.view(*shape[:-1], -1) # [B, ..., 2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        if self.opt.barf_c2f is not None:
            # set weights for different frequency bands
            start, end = self.opt.barf_c2f
            alpha = (self.progress.data - start) / (end - start) * ll
            k = torch.arange(ll, dtype=torch.float32, device=self.opt.device)
            weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1, ll) * weight).view(*shape)
        return input_enc

# ============================ Implicit Mask Generation ============================

class ImplicitMask(torch.nn.Module):
    """Neural Image Function for Masks Generated during training process"""
    def __init__(self, latent=3*128, W=256, in_channels_dir=42):
        super().__init__()
        self.mask_mapping = nn.Sequential(
                            nn.Linear(latent + in_channels_dir, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, 1), nn.Sigmoid())

    def forward(self, x):
        mask = self.mask_mapping(x)
        return mask


class PosEmbedding(torch.nn.Module):
    """Positional Encoding Network"""
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)