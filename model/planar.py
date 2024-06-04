"""Contains Necessary classes for planar BARF"""
import os
import importlib
import time
import numpy as np
import torch
import torch.nn.functional as torch_F
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import PIL
import PIL.Image
import PIL.ImageDraw
import imageio
import visdom

import util
import util_vis
from util import log
import warp

# ============================ main engine for training and evaluation ============================

class Model(torch.nn.Module):
    """DL Model for Planar BARF"""

    def __init__(self, opt):
        super().__init__()
        os.makedirs(opt.output_path,exist_ok=True)
        opt.H_crop, opt.W_crop = opt.data.patch_crop
        # load dataset
        self.image_raw = None
        # build networks
        self.graph = None
        # setup optimizer
        self.optim = None
        self.sched = None
        # restore checkpoint
        self.epoch_start = 0
        self.iter_start = 0

        self.tb = None
        self.vis = None

        self.box_colors = None
        self.vis_path = None
        self.video_fname = None
        self.timer = None
        self.warp_pert = None
        self.ep = self.it = self.vis_it = 0
        # self.vis = None

    def load_dataset(self, opt):
        """Load (single) raw input image into a tensor."""
        log.info("loading dataset...")
        image_raw = PIL.Image.open(opt.data.image_fname)
        self.image_raw = torchvision_F.to_tensor(image_raw).to(opt.device)

    def build_networks(self, opt):
        """Builds Network"""
        log.info("building networks...")
        self.graph = Graph(opt).to(opt.device)

    def setup_optimizer(self, opt):
        """Set up optimizers"""
        log.info("setting up optimizers...")
        optim_list = [
            dict(params=self.graph.neural_image.parameters(), lr=opt.optim.lr),
            dict(params=self.graph.warp_param.parameters(), lr=opt.optim.lr_warp),
        ]
        optimizer = getattr(torch.optim, opt.optim.algo)
        self.optim = optimizer(optim_list)
        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched.type)
            kwargs = { k:v for k, v in opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim, **kwargs)

    def setup_visualizer(self, opt):
        """Setup vis"""
        log.info("setting up visualizers...")
        # tensorboard options
        if opt.tb:
            self.tb = torch.utils.tensorboard.SummaryWriter(log_dir=opt.output_path,flush_secs=10)
        if opt.visdom:
            # check if visdom server is runninng
            is_open = util.check_socket_open(opt.visdom.server,opt.visdom.port)
            retry = None
            while not is_open:
                retry = input(f"visdom port ({opt.visdom.port}) not open, retry? (y/n) ")
                if retry not in ["y","n"]:
                    continue
                if retry=="y":
                    is_open = util.check_socket_open(opt.visdom.server,opt.visdom.port)
                else:
                    break
            self.vis = visdom.Visdom(server=opt.visdom.server,port=opt.visdom.port,env=opt.group)
        # set colors for visualization
        box_colors = ["#ff0000", "#40afff", "#9314ff", "#ffd700", "#00ff00"]
        box_colors = list(map(util.colorcode_to_number, box_colors))
        self.box_colors = np.array(box_colors).astype(int)
        assert len(self.box_colors) == opt.batch_size
        # create visualization directory
        self.vis_path = f"{opt.output_path}/vis"
        os.makedirs(self.vis_path, exist_ok=True)
        self.video_fname = f"{opt.output_path}/vis.mp4"

    def save_checkpoint(self, opt, ep=0, it=0, latest=False):
        """Save Checkpoint"""
        util.save_checkpoint(opt,self,ep=ep,it=it,latest=latest)
        if not latest:
            log.info(
                f"checkpoint saved: ({opt.group}) {opt.name}, epoch {ep} (iteration {it})"
                )

    def restore_checkpoint(self,opt):
        """Restore Checkpoint"""
        epoch_start,iter_start = None,None
        if opt.resume:
            log.info("resuming from previous checkpoint...")
            epoch_start,iter_start = util.restore_checkpoint(opt,self,resume=opt.resume)
        elif opt.load is not None:
            log.info(f"loading weights from checkpoint {opt.load}...")
            epoch_start,iter_start = util.restore_checkpoint(opt,self,load_name=opt.load)
        else:
            log.info("initializing weights from scratch...")
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0

    def train(self, opt):
        """Train Model"""
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(), it_mean=None)
        self.ep = self.it = self.vis_it = 0
        self.graph.train()
        var = edict(idx=torch.arange(opt.batch_size))
        # pre-generate perturbations

        # basically, this function is useless for our further approach.
        # it generates warp perturbations from the 2D image which it
        # tries to restore data from later on.
        self.warp_pert, var.image_pert = self.generate_warp_perturbation(opt)


        # train
        var = util.move_to_device(var, opt.device)
        loader = tqdm.trange(opt.max_iter, desc="training", leave=False)
        # visualize initial state
        var = self.graph.forward(opt, var)
        self.visualize(opt, var, step=0)
        for _ in loader:
            # train iteration
            _ = self.train_iteration(opt, var, loader)
            if opt.warp.fix_first:
                self.graph.warp_param.weight.data[0] = 0


        # after training
        # generate video
        os.system(
            f"ffmpeg -y -framerate 30 -i {self.vis_path}/%d.png -pix_fmt yuv420p {self.video_fname}"
        )
        self.save_checkpoint(opt, ep=None, it=self.it)
        # clear tensorboard and visualization
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom:
            self.vis.close()
        log.title("TRAINING DONE")

    def summarize_loss(self, opt, _, loss):
        """Summarize loss"""
        loss_all = 0.
        assert "all" not in loss
        # weigh losses
        for key in loss:
            assert key in opt.loss_weight
            assert loss[key].shape==()
            if opt.loss_weight[key] is not None:
                assert not torch.isinf(loss[key]), f"loss {key} is Inf"
                assert not torch.isnan(loss[key]), f"loss {key} is NaN"
                loss_all += 10**float(opt.loss_weight[key])*loss[key] # = 10 because render loss = 0
        loss.update(all=loss_all)
        return loss

    def train_iteration(self,opt,var,loader):
        """Train one iteration"""
        # before train iteration
        self.timer.it_start = time.time()
        # train iteration
        self.optim.zero_grad() # reset gradients
        var = self.graph.forward(opt,var,mode="train")
        loss = self.graph.compute_loss(opt,var,mode="train")
        loss = self.summarize_loss(opt,var,loss)
        loss.all.backward()
        self.optim.step()
        # after train iteration
        if (self.it+1)%opt.freq.scalar==0:
            self.log_scalars(opt,var,loss,step=self.it+1,split="train")
        if (self.it+1)%opt.freq.vis==0:
            self.visualize(opt,var,step=self.it+1,split="train")
        self.it += 1
        loader.set_postfix(it=self.it,loss=f"{loss.all:.3f}")
        self.timer.it_end = time.time()
        util.update_timer(opt,self.timer,self.ep,len(loader))
        self.graph.neural_image.progress.data.fill_(self.it/opt.max_iter)
        return loss

    def generate_warp_perturbation(self, opt):
        """generate warp perturbations"""
        # pre-generate perturbations (translational noise + homography noise)
        warp_pert_all = torch.zeros(opt.batch_size, opt.warp.dof, device=opt.device) # 5 x 8

        # noise_t: 0.2
        # noise_h: 0.1

        # (0,0) + 4 corner points
        trans_pert = [(0, 0)]+[(x, y) for x in (-0.2, 0.2) for y in (-0.2, 0.2)]

        def create_random_perturbation():
            warp_pert = torch.randn(opt.warp.dof, device=opt.device) * opt.warp.noise_h # tensor lenght 8 with noise
            # on the first and second value, add the position value of the current batch (see trans_pert)
            # to the first two values of the randomized tensor.
            warp_pert[0] += trans_pert[i][0]
            warp_pert[1] += trans_pert[i][1]
            return warp_pert

        for i in range(opt.batch_size): # i in [0..4]
            warp_pert = create_random_perturbation() # length 8, modified first two values
            while not warp.check_corners_in_range(opt, warp_pert[None]):
                warp_pert = create_random_perturbation()
            warp_pert_all[i] = warp_pert
        if opt.warp.fix_first:
            warp_pert_all[0] = 0
        # create warped image patches
        xy_grid = warp.get_normalized_pixel_grid_crop(opt) # [B, HW, 2]
        xy_grid_warped = warp.warp_grid(opt, xy_grid, warp_pert_all)
        xy_grid_warped = xy_grid_warped.view([opt.batch_size, opt.H_crop, opt.W_crop, 2])
        xy_grid_warped = torch.stack([xy_grid_warped[..., 0]*max(opt.H, opt.W)/opt.W,
                                      xy_grid_warped[..., 1]*max(opt.H, opt.W)/opt.H], dim=-1)
        image_raw_batch = self.image_raw.repeat(opt.batch_size, 1, 1, 1)
        image_pert_all = torch_F.grid_sample(image_raw_batch, xy_grid_warped, align_corners=False)
        return warp_pert_all, image_pert_all

    def visualize_patches(self, opt, warp_param):
        """"Visualize patches"""
        image_pil = torchvision_F.to_pil_image(self.image_raw).convert("RGBA")
        draw_pil = PIL.Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(draw_pil)
        corners_all = warp.warp_corners(opt, warp_param)
        corners_all[..., 0] = (corners_all[..., 0]/opt.W*max(opt.H, opt.W)+1)/2*opt.W-0.5
        corners_all[..., 1] = (corners_all[..., 1]/opt.H*max(opt.H, opt.W)+1)/2*opt.H-0.5
        for i, corners in enumerate(corners_all):
            p = [tuple(float(n) for n in corners[j]) for j in range(4)]
            draw.line([p[0], p[1], p[2], p[3], p[0]], fill=tuple(self.box_colors[i]), width=3)
        image_pil.alpha_composite(draw_pil)
        image_tensor = torchvision_F.to_tensor(image_pil.convert("RGB"))
        return image_tensor

    # @torch.no_grad()
    # def validate(self,opt,ep=None):
    #     self.graph.eval()
    #     loss_val = edict()
    #     loader = tqdm.tqdm(self.test_loader,desc="validating",leave=False)
    #     for it,batch in enumerate(loader):
    #         var = edict(batch)
    #         var = util.move_to_device(var,opt.device)
    #         var = self.graph.forward(opt,var,mode="val")
    #         loss = self.graph.compute_loss(opt,var,mode="val")
    #         loss = self.summarize_loss(opt,var,loss)
    #         for key in loss:
    #             loss_val.setdefault(key,0.)
    #             loss_val[key] += loss[key]*len(var.idx)
    #         loader.set_postfix(loss="{:.3f}".format(loss.all))
    #         if it==0: self.visualize(opt,var,step=ep,split="val")
    #     for key in loss_val: loss_val[key] /= len(self.test_data)
    #     self.log_scalars(opt,var,loss_val,step=ep,split="val")
    #     log.loss_val(opt,loss_val.all)

    @torch.no_grad()
    def predict_entire_image(self, opt):
        """Predict entire image"""
        xy_grid = warp.get_normalized_pixel_grid(opt)[:1]
        rgb = self.graph.neural_image.forward(opt, xy_grid) # [B, HW, 3]
        image = rgb.view(opt.H, opt.W, 3).detach().cpu().permute(2, 0, 1)
        return image

    @torch.no_grad()
    def log_scalars(self, opt, _, loss, metric=None, step=0, split="train"):
        """log scalars"""
        for key,value in loss.items():
            if key=="all":
                continue
            if opt.loss_weight[key] is not None:
                self.tb.add_scalar(f"{split}/loss_{key}",value,step)
        if metric is not None:
            for key,value in metric.items():
                self.tb.add_scalar(f"{split}/{key}",value,step)
        # compute PSNR
        psnr = -10*loss.render.log10()
        self.tb.add_scalar(f"{split}/PSNR", psnr, step)
        # warp error
        warp_error = (self.graph.warp_param.weight-self.warp_pert).norm(dim=-1).mean()
        self.tb.add_scalar(f"{split}/warp error", warp_error, step)

    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train"):
        """vizualize"""
        # dump frames for writing to video
        frame_gt = self.visualize_patches(opt, self.warp_pert)
        frame = self.visualize_patches(opt, self.graph.warp_param.weight)
        frame2 = self.predict_entire_image(opt)
        frame_cat = (torch.cat([frame, frame2], dim=1)*255).byte().permute(1, 2, 0).numpy()
        imageio.imsave(f"{self.vis_path}/{self.vis_it}.png", frame_cat)
        self.vis_it += 1
        # visualize in Tensorboard
        if opt.tb:
            colors = self.box_colors
            util_vis.tb_image(
                opt, self.tb, step, split, "image_pert", util_vis.color_border(var.image_pert, colors) # pylint: disable=line-too-long
                )
            util_vis.tb_image(
                opt, self.tb, step, split, "rgb_warped", util_vis.color_border(var.rgb_warped_map, colors) # pylint: disable=line-too-long
                )
            util_vis.tb_image(
                opt, self.tb, self.it+1, "train", "image_boxes", frame[None]
                )
            util_vis.tb_image(
                opt, self.tb, self.it+1, "train", "image_boxes_GT", frame_gt[None]
                )
            util_vis.tb_image(
                opt, self.tb, self.it+1, "train", "image_entire", frame2[None]
                )

# ============================ computation graph for forward/backprop ============================

class Graph(torch.nn.Module):
    """Graph for planar BARF"""

    def __init__(self, opt):
        super().__init__()
        self.neural_image = NeuralImageFunction(opt)
        self.graph.warp_param = torch.nn.Embedding(opt.batch_size, opt.warp.dof).to(opt.device)
        torch.nn.init.zeros_(self.graph.warp_param.weight)

    def forward(self, opt, var, mode=None): # pylint: disable=unused-argument
        """Forward graph"""
        xy_grid = warp.get_normalized_pixel_grid_crop(opt)
        xy_grid_warped = warp.warp_grid(opt, xy_grid, self.warp_param.weight)
        # render images
        var.rgb_warped = self.neural_image.forward(opt, xy_grid_warped) # [B, HW, 3]
        var.rgb_warped_map = var.rgb_warped.view(
            opt.batch_size, opt.H_crop, opt.W_crop, 3
            ).permute(0, 3, 1, 2) # [B, 3, H, W]
        return var

    def compute_loss(self, opt, var, mode=None): # pylint: disable=unused-argument
        """Compute Loss"""
        loss = edict()
        if opt.loss_weight.render is not None:
            image_pert = var.image_pert.view(opt.batch_size, 3, opt.H_crop*opt.W_crop).permute(0, 2, 1) # pylint: disable=line-too-long

            # loss gets computed for difference between 
            # - previously calculated image patch (ground truth, var.image_pert)
            # - current learned estimation (var.rgb_warped)
            loss.render = self.mse_loss(var.rgb_warped, image_pert)
        return loss

    def l1_loss(self,pred,label=0):
        """L1 Loss function"""
        loss = (pred.contiguous()-label).abs()
        return loss.mean()
    def mse_loss(self,pred,label=0):
        """MSE Loss Function"""
        loss = (pred.contiguous()-label)**2
        return loss.mean()


# ============================ Neural Image Function ============================

class NeuralImageFunction(torch.nn.Module):
    """Neural Image Function for planar BARF"""

    def __init__(self, opt):
        super().__init__()
        self.define_network(opt)
        self.progress = torch.nn.Parameter(
            torch.tensor(0.)
            ) # use Parameter so it could be checkpointed

    def define_network(self, opt):
        """Define Network"""
        input_2d_dim = 2+4*opt.arch.posenc.L_2D if opt.arch.posenc else 2
        # point-wise RGB prediction
        self.mlp = torch.nn.ModuleList()
        ll = util.get_layer_dims(opt.arch.layers)
        for li, (k_in, k_out) in enumerate(ll):
            if li == 0:
                k_in = input_2d_dim
            if li in opt.arch.skip:
                k_in += input_2d_dim
            linear = torch.nn.Linear(k_in, k_out)
            if opt.barf_c2f and li==0:
                # rescale first layer init (distribution was for pos.enc. but only xy is first used)
                scale = np.sqrt(input_2d_dim/2.)
                linear.weight.data *= scale
                linear.bias.data *= scale
            self.mlp.append(linear)

    def forward(self, opt, coord_2d): # [B, ..., 3]
        """Forward through Network"""
        if opt.arch.posenc:
            points_enc = self.positional_encoding(opt, coord_2d, ll=opt.arch.posenc.L_2D)
            points_enc = torch.cat([coord_2d, points_enc], dim=-1) # [B, ..., 6L+3]
        else:
            points_enc = coord_2d
        feat = points_enc
        # extract implicit features
        for li, layer in enumerate(self.mlp):
            if li in opt.arch.skip:
                feat = torch.cat([feat, points_enc], dim=-1)
            feat = layer(feat)
            if li!=len(self.mlp)-1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid_() # [B, ..., 3]
        return rgb

    def positional_encoding(self, opt, input_img, ll): # [B, ..., N]
        """PosEnc Layers"""
        shape = input_img.shape
        freq = 2**torch.arange(ll, dtype=torch.float32, device=opt.device)*np.pi # [ll]
        spectrum = input_img[..., None]*freq # [B, ..., N, L]
        sin, cos = spectrum.sin(), spectrum.cos() # [B, ..., N, L]
        input_enc = torch.stack([sin, cos], dim=-2) # [B, ..., N, 2, L]
        input_enc = input_enc.view(*shape[:-1], -1) # [B, ..., 2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        if opt.barf_c2f is not None:
            # set weights for different frequency bands
            start, end = opt.barf_c2f
            alpha = (self.progress.data-start)/(end-start)*ll
            k = torch.arange(ll, dtype=torch.float32, device=opt.device)
            weight = (1-(alpha-k).clamp_(min=0, max=1).mul_(np.pi).cos_())/2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1, ll)*weight).view(*shape)
        return input_enc
