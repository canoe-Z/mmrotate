# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox.transforms import norm_angle
from .distance_angle_point_coder import DistanceAnglePointCoder


@TASK_UTILS.register_module()
class DistanceAnglePointCoderEdgeSwap(DistanceAnglePointCoder):
    """Distance Angle Point BBox coder.

    This coder encodes gt bboxes (x, y, w, h, theta) into (top, bottom, left,
    right, theta) and decode it back to the original.

    Args:
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def distance2obb(self,
                     points,
                     distance,
                     max_shape=None,
                     angle_version='oc'):
        distance, angle = distance.split([4, 1], dim=-1)

        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)

        rot_matrix = torch.cat([cos_angle, -sin_angle, sin_angle, cos_angle],
                               dim=-1)
        rot_matrix = rot_matrix.reshape(*rot_matrix.shape[:-1], 2, 2)

        wh = distance[..., :2] + distance[..., 2:]
        offset_t = (distance[..., 2:] - distance[..., :2]) / 2
        offset = torch.matmul(rot_matrix, offset_t[..., None]).squeeze(-1)
        ctr = points[..., :2] + offset

        angle_regular = norm_angle(angle, angle_version)

        # egde_swap
        gw, gh = wh.split([1, 1], dim=-1)
        gw_regular = torch.where(gw > gh, gw, gh)
        gh_regular = torch.where(gw > gh, gh, gw)
        angle_regular = torch.where(
            gw > gh, angle_regular, angle_regular + np.pi / 2)
        angle_regular = norm_angle(angle_regular, angle_version)
        return torch.cat([ctr, gw_regular, gh_regular, angle_regular], dim=-1)
