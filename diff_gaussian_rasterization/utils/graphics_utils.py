#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrix2(znear, zfar, K, W, H):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    top = znear * cy / fy
    bottom = -znear * (H - cy) / fy
    right = znear * (W - cx) / fx
    left = -znear * cx / fx

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = -(right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def getPerspectiveMatrix(znear, zfar, K, W, H):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    top = -znear * cy / fy
    bottom = znear * (H - cy) / fy
    right = znear * (W - cx) / fx
    left = -znear * cx / fx

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = -2.0 * znear / (top - bottom)
    P[0, 2] = -(right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def getOrthographicMatrix(znear, zfar, top, bottom, right, left):
    """
    Get orthographic projection matrix.，全部是相对于相机坐标系的坐标， x->right, y-down, z->forward
    Args:
        znear:
        zfar:
        top:
        bottom:
        right:
        left:

    Returns:

    """
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 / (right - left)
    P[0, 3] = -(right + left) / (right - left)
    P[1, 1] = 2.0 / (bottom - top)
    P[1, 3] = -1.0 * (top + bottom) / (bottom - top)
    P[2, 2] = 1.0 / (zfar - znear)
    P[2, 3] = -znear / (zfar - znear)
    P[3, 3] = 1.0
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))
