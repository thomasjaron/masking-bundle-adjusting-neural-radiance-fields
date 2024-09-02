"""Module representing warps"""
# pylint: disable=invalid-name
import torch

class Warp:
    """"Utility functions to perform warps and other transformations for image
    preparation or retrieval"""

    def __init__(self, opt):
        self.max_h = opt.H
        self.crop_h = opt.patch_H
        self.max_w = opt.W
        self.crop_w = opt.patch_W
        self.y_crop = (
            self.max_h // 2 - self.crop_h // 2, self.max_h // 2 + self.crop_h // 2
            )
        self.x_crop = (
            self.max_w // 2 - self.crop_w // 2, self.max_w // 2 + self.crop_w // 2
            )
        self.norm_h = self.max_h / max(self.max_h, self.max_w)
        self.norm_w = self.max_w / max(self.max_h, self.max_w)
        self.batch_size = opt.batch_size
        self.device = opt.device
        self.warp_type = opt.warp.type
        self.dof = opt.warp.dof

    def to_hom(self, matrix):
        """Convert a matrix to the homogenous format."""
        # get homogeneous coordinates of the input
        mat_hom = torch.cat([matrix, torch.ones_like(matrix[..., :1])], dim=-1)
        return mat_hom

    def get_normalized_pixel_grid(self, crop=False):
        """Create a pixel grid fitting to the output image width and height,
        which optionally can be cropped to fit the input image width and height"""
        # prepare grid dimensions
        if crop:
            y_range = (
                (torch.arange(
                    *(self.y_crop),
                    dtype=torch.float32,
                    device=self.device) + 0.5
                ) / self.max_h * 2 - 1) * self.norm_h
            x_range = (
                (torch.arange(
                    *(self.x_crop),
                    dtype=torch.float32,
                    device=self.device) + 0.5
                ) / self.max_w * 2 - 1) * self.norm_w
            Y, X = torch.meshgrid(y_range, x_range)  # [H, W]
            xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW, 2]
            xy_grid = xy_grid.repeat(self.batch_size, 1, 1)  # [B, HW, 2]
            return xy_grid
        else:
            y_range = ((torch.arange(
                    self.max_h,
                    dtype=torch.float32,
                    device=self.device) + 0.5
                ) / self.max_h * 2 - 1) * self.norm_h
            x_range = ((torch.arange(
                    self.max_w,
                    dtype=torch.float32,
                    device=self.device) + 0.5
                ) / self.max_w * 2 - 1) * self.norm_w
            Y, X = torch.meshgrid(y_range, x_range)  # [H, W]
            xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW, 2]
            xy_grid = xy_grid.repeat(self.batch_size, 1, 1)  # [B, HW, 2]
            return xy_grid

    def warp_grid(self, xy_grid, warp):
        """Depending on given options, perform a transformation onto a grid or image."""
        if self.warp_type == "homography":
            assert self.dof == 8
            xy_grid_hom = self.to_hom(xy_grid)
            warp_matrix = lie.sl3_to_SL3(warp)
            warped_grid_hom = xy_grid_hom@warp_matrix.transpose(-2, -1)
            warped_grid = warped_grid_hom[..., :2] / \
                (warped_grid_hom[..., 2:]+1e-8)  # [B,HW,2]
        else:
            assert False
        return warped_grid

    def warp_corners(self, warp_param):
        """Helper function to visualize estimated homographies for input images."""
        # use the two range values and generate two new values representing the corners
        Y = [((y + 0.5) / self.max_h * 2 - 1) * (self.norm_h) for y in self.y_crop]
        X = [((x + 0.5) / self.max_w * 2 - 1) * (self.norm_w) for x in self.x_crop]
        corners = [(X[0], Y[0]), (X[0], Y[1]), (X[1], Y[1]), (X[1], Y[0])] # [4]
        # create a matrix with 5 copies of the corners tensor
        corners = torch.tensor(corners, dtype=torch.float32,
                            device=self.device).repeat(self.batch_size, 1, 1) # [B, 4]
        corners_warped = self.warp_grid(corners, warp_param)
        return corners_warped

class Lie():
    """Linear algebra functions"""

    def sl3_to_SL3(self, h):
        """homography: directly expand matrix exponential"""
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.6151&rep=rep1&type=pdf
        h1, h2, h3, h4, h5, h6, h7, h8 = h.chunk(8, dim=-1)
        A = torch.stack([torch.cat([h5, h3, h1], dim=-1),
                         torch.cat([h4, -h5-h6, h2], dim=-1),
                         torch.cat([h7, h8, h6], dim=-1)], dim=-2)
        H = A.matrix_exp()
        return H

lie = Lie()
