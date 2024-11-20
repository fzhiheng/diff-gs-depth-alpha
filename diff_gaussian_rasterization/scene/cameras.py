from typing import List

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
        self.top = top
        self.bottom = bottom
        self.right = right
        self.left = left
        self.device = torch.device(device)
        self.world_view_transform = torch.linalg.inv(cam2world).transpose(0, 1).to(device)
        self.projection_matrix = getOrthographicMatrix(znear, zfar, top, bottom, right, left).transpose(0, 1).to(device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def __repr__(self):
        return f"OrthographicCamera[W:{self.image_width}, H:{self.image_height}, znear:{self.znear}, zfar:{self.zfar}, top:{self.top}, bottom:{self.bottom}, right:{self.right}, left:{self.left}]"

    def split(self, *, edge_length: float = None, edge_pixel: int = None) -> List[List['OrthographicCamera']]:
        """Split the camera into multiple cameras with the edge length or edge pixel

        Returns:
            _type_: List[List['OrthographicCamera']], from top to bottom, left to right
        """
        if edge_length is None and edge_pixel is None:
            raise ValueError("You must provide either edge_length or edge_pixel")
        if edge_length is not None and edge_pixel is not None:
            raise ValueError("You must provide either edge_length or edge_pixel, not both")
        tb_length = abs(self.bottom - self.top)
        rl_length = abs(self.right - self.left)
        if edge_length is not None:
            if tb_length <= edge_length and rl_length <= edge_length:
                return [[self]]
            if tb_length > rl_length:
                edge_pixel = int(self.image_height * edge_length / tb_length)
                real_edge = tb_length * edge_pixel / self.image_height
            else:
                edge_pixel = int(self.image_width * edge_length / rl_length)
                real_edge = rl_length * edge_pixel / self.image_width
        else:
            if self.image_height <= edge_pixel and self.image_width <= edge_pixel:
                return [[self]]
            if self.image_height > self.image_width:
                real_edge = edge_pixel / self.image_height * tb_length
            else:
                real_edge = edge_pixel / self.image_width * rl_length

        tb_num = int(self.image_height / edge_pixel)
        rl_num = int(self.image_width / edge_pixel)
        heigths = [edge_pixel] * tb_num
        widths = [edge_pixel] * rl_num
        tbs = [self.top + real_edge * i for i in range(0, tb_num + 1)]
        lrs = [self.left + real_edge * i for i in range(0, rl_num + 1)]

        if edge_pixel * tb_num < self.image_height:
            heigths.append(self.image_height - edge_pixel * tb_num)
            tbs.append(self.bottom)
        if edge_pixel * rl_num < self.image_width:
            widths.append(self.image_width - edge_pixel * rl_num)
            lrs.append(self.right)
        tops, bottoms = tbs[:-1], tbs[1:]
        lefts, rights = lrs[:-1], lrs[1:]

        cameras = []
        for t, b, h in zip(tops, bottoms, heigths):
            cameras.append([])
            for l, r, w in zip(lefts, rights, widths):
                cameras[-1].append(
                    OrthographicCamera(self.cam2world[:3, :3], self.cam2world[:3, 3], w, h, self.znear, self.zfar, t, b, r, l, self.device))
        return cameras