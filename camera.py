"""Module representing a Camera"""

# pylint: disable=invalid-name
import torch

class Pose():
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [r|t]
    """

    def __call__(self, r=None, t=None):
        # construct a camera pose from the given r and/or t
        assert (r is not None or t is not None)
        if r is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            r = torch.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(r, torch.Tensor):
                r = torch.tensor(r)
            t = torch.zeros(r.shape[:-1], device=r.device)
        else:
            if not isinstance(r, torch.Tensor):
                r = torch.tensor(r)
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
        assert (r.shape[:-1] == t.shape and r.shape[-2:] == (3, 3))
        r = r.float()
        t = t.float()
        p = torch.cat([r, t[..., None]], dim=-1)  # [...,3,4]
        assert p.shape[-2:] == (3, 4)
        return p

    def invert(self, p, use_inverse=False):
        """Docstring"""
        # invert a camera pose
        r, t = p[..., :3], p[..., 3:]
        r_inv = r.inverse() if use_inverse else r.transpose(-1, -2)
        t_inv = (-r_inv @ t)[..., 0]
        pose_inv = self(r=r_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        """Docstring"""
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for p in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, p)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        """Docstring"""
        # pose_new(x) = pose_b o pose_a(x)
        r_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        r_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        r_new = r_b@r_a
        t_new = (r_b@t_a+t_b)[..., 0]
        pose_new = self(r=r_new, t=t_new)
        return pose_new

pose = Pose()

def to_hom(X):
    """Docstring"""
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom

def cam2world(X, p):
    """Docstring"""
    X_hom = to_hom(X)
    pose_inv = Pose().invert(p)
    return X_hom@pose_inv.transpose(-1, -2)
