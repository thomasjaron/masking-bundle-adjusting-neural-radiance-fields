"""Contains Necessary classes for planar BARF"""
import os
import time
import numpy as np
import torch
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

# ============================ main engine for training and evaluation ============================

class Model(torch.nn.Module):
    """DL Model for Planar BARF"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.dataset = opt.dataset
        os.makedirs(opt.output_path,exist_ok=True)
        self.warp = Warp(opt)
        # container for all images and image processings
        self.images = None
        # build networks
        self.graph = None
        # setup optimizer
        self.optim = None
        self.sched = None
        # restore checkpoint
        self.epoch_start = 0
        self.iter_start = 0

        self.tb = None

        self.box_colors = None
        self.vis_path = None
        self.video_fname = None
        self.timer = None
        self.warp_pert = None
        self.ep = self.it = self.vis_it = 0

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
            fps_masks=mask_paths if self.opt.use_masks else None,
            fp_gt=f'data/planar/{self.dataset}/gt.png',
            edges = True if self.opt.use_edges else None
            )
        # plt.figure()
        # plt.imshow(images.edges)
        # plt.title('Original Image')
        # plt.axis('off')
        # plt.show()

    def build_networks(self):
        """Builds Network"""
        log.info("building networks...")
        self.graph = Graph(self.opt).to(self.opt.device)

    def setup_optimizer(self):
        """Set up optimizers"""
        log.info("setting up optimizers...")
        optim_list = [
            dict(params=self.graph.neural_image.parameters(), lr=self.opt.optim.lr),
            dict(params=self.graph.warp_param.parameters(), lr=self.opt.optim.lr_warp),
        ]
        optimizer = getattr(torch.optim, self.opt.optim.algo)
        self.optim = optimizer(optim_list)
        # set up scheduler
        if self.opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler, self.opt.optim.sched.type)
            kwargs = { k:v for k, v in self.opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim, **kwargs)

    def setup_visualizer(self):
        """Setup vis"""
        log.info("setting up visualizers...")
        # activate tensorboard
        if self.opt.tb:
            self.tb = torch.utils.tensorboard.SummaryWriter(log_dir=self.opt.output_path,flush_secs=10)

        # Prepare homography visualization
        box_colors = [
            "#FF5733",  # Red-Orange
            "#33FF57",  # Lime Green
            "#3357FF",  # Blue
            "#FF33A6",  # Pink
            "#FF8C33",  # Orange
            "#8C33FF",  # Purple
            "#33FFF0",  # Aqua
            "#FF33F0",  # Magenta
            "#33FF8C",  # Mint Green
            "#FF3333",  # Red
            "#FFFF33",  # Yellow
            "#33FFFF",  # Cyan
            "#5733FF",  # Indigo
            "#33FF57",  # Light Green
            "#FF5733",  # Coral
            "#FF33FF",  # Fuchsia
            "#57FF33",  # Spring Green
            "#5733FF",  # Blue Violet
            "#FF3357",  # Reddish Pink
            "#8CFF33"   # Chartreuse
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
        """Summarize loss"""
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
        if (self.it+1) % self.opt.freq.scalar==0:
            if self.tb:
                self.log_scalars(loss, step=self.it+1, split="train")
        if (self.it+1) % self.opt.freq.vis==0:
            self.visualize(var, step=self.it+1, split="train")
        self.it += 1
        loader.set_postfix(it=self.it, loss=f"{loss.all:.3f}")
        self.timer.it_end = time.time()
        util.update_timer(self.opt, self.timer, self.ep, len(loader))
        self.graph.neural_image.progress.data.fill_(self.it / self.opt.max_iter)
        return loss

    def visualize_patches(self, warp_param):
        """"Visualize current homography estimation for each image"""
        image_pil = torchvision_F.to_pil_image(self.images.gt).convert("RGBA")
        draw_pil = PIL.Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(draw_pil)
        # compute corners of each homography
        corners_all = self.warp.warp_corners(warp_param)
        corners_all[..., 0] = (
            corners_all[..., 0] / self.opt.W * max(self.opt.H, self.opt.W) + 1
            ) / 2 * self.opt.W - 0.5
        corners_all[..., 1] = (
            corners_all[..., 1] / self.opt.H * max(self.opt.H, self.opt.W) + 1
            ) / 2 * self.opt.H - 0.5
        # draw warped squares of each patch onto canvas
        for i, corners in enumerate(corners_all):
            p = [tuple(float(n) for n in corners[j]) for j in range(4)]
            draw.line([p[0], p[1], p[2], p[3], p[0]], fill=tuple(self.box_colors[i]), width=3)
        # merge canvas with image (ground truth image)
        image_pil.alpha_composite(draw_pil)
        image_tensor = torchvision_F.to_tensor(image_pil.convert("RGB"))
        return image_tensor

    @torch.no_grad()
    def validate(self,opt,ep=None):
        """docstring"""
        self.graph.eval()
        loss_val = edict()
        loader = tqdm.tqdm(self.test_loader,desc="validating",leave=False)
        for it,batch in enumerate(loader):
            var = edict(batch)
            var = util.move_to_device(var,opt.device)
            var = self.graph.forward(opt,var,mode="val")
            loss = self.graph.compute_loss(opt,var,mode="val")
            loss = self.summarize_loss(loss)
            for key in loss:
                loss_val.setdefault(key,0.)
                loss_val[key] += loss[key]*len(var.idx)
            loader.set_postfix(loss=f"{loss.all:.3f}")
            if it==0:
                self.visualize(var,step=ep,split="val")
        for key in loss_val:
            loss_val[key] /= len(self.test_data)
        if self.tb:
            self.log_scalars(opt,loss_val,step=ep,split="val")
        # log.loss_val(loss_val.all)

    @torch.no_grad()
    def predict_entire_image(self):
        """Predict entire image"""
        xy_grid = self.warp.get_normalized_pixel_grid()[:1]
        rgb = self.graph.neural_image.forward(xy_grid) # [B, HW, 3]
        image = rgb.view(self.opt.H, self.opt.W, 3).detach().cpu().permute(2, 0, 1)
        return image

    @torch.no_grad()
    def log_scalars(self, loss, metric=None, step=0, split="train"):
        """log scalars"""
        for key,value in loss.items():
            if key=="all":
                continue
            if self.opt.loss_weight[key] is not None:
                self.tb.add_scalar(f"{split}/loss_{key}",value,step)
        if metric is not None:
            for key,value in metric.items():
                self.tb.add_scalar(f"{split}/{key}",value,step)
        # compute PSNR
        psnr = -10 * loss.render.log10()
        self.tb.add_scalar(f"{split}/PSNR", psnr, step)

    @torch.no_grad()
    def visualize(self, var, step=0, split="train"):
        """vizualize"""
        # dump frames for writing to video
        frame = self.visualize_patches(self.graph.warp_param.weight) # upper image
        frame2 = self.predict_entire_image() # prediction, lower image
        # vertically align the images and cast them to valid values [0...255]
        frame_cat = (torch.cat([frame, frame2], dim=1)*255).byte().permute(1, 2, 0).numpy()
        imageio.imsave(f"{self.vis_path}/{self.vis_it}.png", frame_cat)
        # for indexx, p in enumerate(var.rgb_prediction_map):
        #     pic = p.cpu().permute(1, 2, 0).numpy()
        #     imageio.imsave(f"{self.vis_path}/rgbwarped-{self.vis_it}-{indexx}.png", (pic * 255).astype(np.uint8))
        self.vis_it += 1
        # visualize in Tensorboard
        if self.opt.tb:
            colors = self.box_colors
            util_vis.tb_image( # [B, 3, H, W]
                self.opt, self.tb, self.it+1, "train", "image_pert", util_vis.color_border(var.images.rgb, colors) # pylint: disable=line-too-long
                )
            util_vis.tb_image(
                self.opt, self.tb, self.it+1, "train", "rgb_warped", util_vis.color_border((var.rgb_prediction_map*255), colors) # pylint: disable=line-too-long
                )
            util_vis.tb_image(
                self.opt, self.tb, self.it+1, "train", "image_boxes", frame[None]
                )
            # util_vis.tb_image(
            #     self.opt, self.tb, self.it+1, "train", "image_boxes_GT", frame_gt[None]
            #     )
            util_vis.tb_image(
                self.opt, self.tb, self.it+1, "train", "image_entire", frame2[None]
                )

# ============================ computation graph for forward/backprop ============================

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

    def forward(self, var, mode=None): # pylint: disable=unused-argument
        """Get image prediction given the current homographies"""
        xy_grid = self.warp.get_normalized_pixel_grid(crop=self.opt.use_cropped_images)
        # warp grid according to warp_param.weight homographies for each image.
        xy_grid_warped = self.warp.warp_grid(xy_grid, self.warp_param.weight)
        # get rgb image prediction for the warped 2d area
        var.rgb_prediction = self.neural_image.forward(xy_grid_warped) # [B, HW, 3]
        #print(var.rgb_prediction.size())
        # used for tensorboard visualisation
        var.rgb_prediction_map = var.rgb_prediction.view(self.batch_size, int(self.h), int(self.w), 3).permute(0, 3, 1, 2) # [B, 3, H, W]
        #print(var.rgb_prediction_map.size())
        return var

    def compute_loss(self, var, mode=None): # pylint: disable=unused-argument
        """Compute Loss"""
        loss = edict()
        if self.opt.loss_weight.render is not None:
            image_pert = var.images.rgb.view(self.batch_size, 3, int(self.h * self.w)).permute(0, 2, 1) # pylint: disable=line-too-long
            mask_pert = None
            edge_pert = None
            if var.images.masks is not None:
                mask_pert = var.images.masks.view(self.batch_size, 1, int(self.h * self.w)).permute(0, 2, 1) # pylint: disable=line-too-long
            if var.images.edges is not None:
                edge_pert = var.images.edges.view(self.batch_size, 1, int(self.h * self.w)).permute(0, 2, 1) # pylint: disable=line-too-long 
            loss.render = self.rgb_loss(var.rgb_prediction, image_pert, mask_pert, edge_pert)
             
        return loss

    def rgb_loss(self, pred, labels, masks=None, edge=None):
        """Perform MSE on RGB color values and use masks if available, including edge loss."""
        alpha = 0.2

        # RGB Loss
        if masks is None:
            rgb_loss = (pred.contiguous() - labels) ** 2
            rgb_loss = rgb_loss.mean()
        else:
            masked_diff = (pred.contiguous() - labels) * masks
            masked_loss = masked_diff ** 2
            rgb_loss = masked_loss.sum() / masks.sum()  # Only average over unmasked elements
        # Edge Loss
        edge_loss = 0
        if edge is not None:
            # height, width = self.h, self.w #uncomment for plt.imshow
            #Change Shape of Prediction and Masks to use compute_edge function
            pred_reshaped = pred.view(self.batch_size, self.h, self.w, 3).permute(0, 3, 1, 2)  # Reshape to [B, 3, H*W]
            mask_reshaped = masks.view(self.batch_size, self.h, self.w, 1).permute(0, 3, 1, 2)  # Reshape to [B, 3, H*W]

            for i in range(len(pred)):  # Iterate over the length of pred
                single_pred = pred_reshaped[i].unsqueeze(0)  # Add batch dimension back
                pred_edges = inputs.compute_edges(single_pred, self.opt.device)  # Compute edges for single_pred
                single_mask = mask_reshaped[i].unsqueeze(0)  # Add batch dimension back
                mask_edges = inputs.compute_edges(single_mask, self.opt.device)  # Compute edges for single_pred

                for j in range(len(edge)):
                    single_edge = edge[j].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions back
                    pred_edges_resized = torch_F.interpolate(pred_edges, size=single_edge.shape[2:], mode='bilinear', align_corners=False)
                    edge_diff = (pred_edges_resized - single_edge) ** 2

                    """
                    VISUALIZING SUBTRACTED EDGES
                    edge_diff_np = edge_diff.detach().cpu().numpy().squeeze()
                    if edge_diff_np.ndim == 4:
                        edge_diff_np = edge_diff_np[0, 0]  # Select the first batch and channel
                    if edge_diff_np.ndim == 1:
                        edge_diff_np = edge_diff_np.reshape(height, width)
                    print(f'edge_diff final shape for imshow: {edge_diff_np.shape}')
                    plt.figure()
                    plt.imshow(edge_diff_np, cmap='gray')
                    plt.title('Edge Difference')
                    plt.axis('off')
                    plt.show()
                    image = edge[j].detach().cpu().numpy(
                    if image.ndim == 3:
                        image = image.squeeze()
                    plt.figure()
                    plt.imshow(image, cmap='gray')
                    plt.title('Edge Image')
                    plt.axis('off')
                    plt.show()
                    """


                    edge_loss += edge_diff.mean()
            edge_loss /= (len(pred) * len(edge))  # Average the edge loss over the batch and edge length
        total_loss = rgb_loss + alpha * edge_loss
        return total_loss

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
        freq = 2**torch.arange(ll, dtype=torch.float32, device=self.opt.device)*np.pi # [ll]
        spectrum = coord_2d[..., None]*freq # [B, ..., N, L]
        sin, cos = spectrum.sin(), spectrum.cos() # [B, ..., N, L]
        input_enc = torch.stack([sin, cos], dim=-2) # [B, ..., N, 2, L]
        input_enc = input_enc.view(*shape[:-1], -1) # [B, ..., 2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        if self.opt.barf_c2f is not None:
            # set weights for different frequency bands
            start, end = self.opt.barf_c2f
            alpha = (self.progress.data-start)/(end-start)*ll
            k = torch.arange(ll, dtype=torch.float32, device=self.opt.device)
            weight = (1-(alpha-k).clamp_(min=0, max=1).mul_(np.pi).cos_())/2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1, ll)*weight).view(*shape)
        return input_enc
