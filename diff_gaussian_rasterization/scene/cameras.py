import torch
from torch import nn

from ..utils.graphics_utils import getOrthographicMatrix, getPerspectiveMatrix, focal2fov

# All camear coordinate systems are RDF (x-right y-down z-forward)

class PerspectiveCamera(nn.Module):
    def __init__(self, R: torch.Tensor, T: torch.Tensor, K, W, H, znear, zfar, device):
        super().__init__()
        self.image_width = W
        self.image_height = H
        self.FoVy = focal2fov(K[1, 1], H)
        self.FoVx = focal2fov(K[0, 0], W)
        cam2world = torch.eye(4, device=device)
        cam2world[:3, :3] = R.to(device)
        cam2world[:3, 3] = T.to(device)
        self.cam2world = cam2world
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = torch.linalg.inv(cam2world).transpose(0, 1)
        self.projection_matrix = getPerspectiveMatrix(znear, zfar, K, W, H).transpose(0, 1).to(device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class OrthographicCamera(nn.Module):
    def __init__(self, R: torch.Tensor, T: torch.Tensor, W, H, znear, zfar, top, bottom, right, left, device):
        super().__init__()
        self.image_width = W
        self.image_height = H
        cam2world = torch.eye(4, device=device)
        cam2world[:3, :3] = R.to(device)
        cam2world[:3, 3] = T.to(device)
        self.cam2world = cam2world
        self.znear = znear
        self.zfar = zfar
        self.data_device = torch.device(device)
        self.world_view_transform = torch.linalg.inv(cam2world).transpose(0, 1).to(device)
        self.projection_matrix = getOrthographicMatrix(znear, zfar, top, bottom, right, left).transpose(0, 1).to(device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
