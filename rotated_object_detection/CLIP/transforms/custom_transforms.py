# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import List, Optional, Union

import cv2
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from torch.distributions.multivariate_normal import MultivariateNormal

from mmrotate.registry import TRANSFORMS

VIS = False
    
@TRANSFORMS.register_module()
class GetOriImg:
    """ use sobel to get edge map from image"""
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, results: dict):
        results['ori_img'] = results['img'].copy()
        return results
    
    def __repr__(self):
        return self.__class__.__name__
    
@TRANSFORMS.register_module()
class Rbox2SegMask:
    """Generate semantic segmentation map ground-truth.
    """

    def __init__(self, cls_num=15, mask_scale_factor=1/4, box_scale_factor=1.0) -> None:
        super(Rbox2SegMask, self).__init__() 
        self.cls_num = cls_num
        self.mask_scale_factor = mask_scale_factor
        self.box_scale_factor = box_scale_factor

    def __call__(self, results: dict) -> dict:
        """The transform function."""
        gt_bboxes = results['gt_bboxes'].tensor.clone()
        gt_labels = results['gt_bboxes_labels']

        h, w = results['img'].shape[:2]
        h, w = int(h * self.mask_scale_factor), int(w * self.mask_scale_factor)
        # Initialize the masks
        global_mask = torch.zeros((h, w), dtype=torch.float32)
        class_specific_targets = torch.zeros((self.cls_num, h, w), dtype=torch.float32)

        gt_bboxes[..., :4] *= self.mask_scale_factor
        gt_bboxes[..., 2:4] *= self.box_scale_factor
        polygons = (obb2poly_le90(gt_bboxes)).cpu().numpy()

        gt_polygons_list = [polygons[gt_labels == i] for i in range(self.cls_num)]

        for i, gt_polygons_cls in enumerate(gt_polygons_list):
            if len(gt_polygons_cls) == 0:
                continue
            polygons_list = [gt_polygons_cls[i, :] for i in range(gt_polygons_cls.shape[0])]
            bitmask  = polygons_to_bitmask_cpu(polygons_list, h, w)

            # Update masks
            global_mask += bitmask 
            class_specific_targets[i] += bitmask

        global_mask = global_mask > 1e-8
        for i in range(self.cls_num):
            class_specific_targets[i] = class_specific_targets[i] > 1e-8 
        
        # # Visualize the results
        # plt.imshow(global_mask, cmap="gray")
        # plt.savefig(f"{results['img_id']}-global_mask.png")
        # plt.close()
        # fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        # for i in range(self.cls_num):
        #     row, col = divmod(i, 5)
        #     axes[row, col].imshow(class_specific_targets[i], cmap="gray")
        #     axes[row, col].set_title(f"Class {i}")

        # plt.tight_layout()
        # plt.savefig(f"{results['img_id']}-class_specific_targets.png")
        # plt.close()

        results['rbox2segmask'] = global_mask.float(), class_specific_targets.float()
        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module()
class Rbox2GSSegMask:
    """Generate semantic segmentation map ground-truth.
    """

    def __init__(self, cls_num=15, mask_scale_factor=1/4, box_scale_factor=0.9) -> None:
        super(Rbox2GSSegMask, self).__init__() 
        self.cls_num = cls_num
        self.mask_scale_factor = mask_scale_factor
        self.box_scale_factor = box_scale_factor
        

    def __call__(self, results: dict) -> dict:
        """The transform function."""
        gt_bboxes = results['gt_bboxes'].tensor.clone()
        gt_labels = results['gt_bboxes_labels']

        h, w = results['img'].shape[:2]
        h, w = int(h * self.mask_scale_factor), int(w * self.mask_scale_factor)
        # Initialize the masks
        global_mask = torch.zeros((h, w), dtype=torch.float32)
        class_specific_targets = torch.zeros((self.cls_num, h, w), dtype=torch.float32)

        gt_bboxes[..., :4] *= self.mask_scale_factor
        gt_bboxes[..., 2:4] *= self.box_scale_factor

        polygons = (obb2poly_le90(gt_bboxes)).cpu().numpy()
        gt_polygons_list = [polygons[gt_labels == i] for i in range(self.cls_num)]

        gt_bboxes_list = [gt_bboxes[gt_labels == i] for i in range(self.cls_num)]

        for i, (gt_bboxes_cls, gt_polygons_cls) in enumerate(zip(gt_bboxes_list, gt_polygons_list)):
            if len(gt_bboxes_cls) == 0:
                continue
            gaussian_mask = rbox2GS_mask(gt_bboxes_cls, h, w)
            gaussian_mask = torch.sum(gaussian_mask, dim=0)
            
            threshold = torch.max(gaussian_mask) * 0.3
            gaussian_mask[gaussian_mask <= threshold] = 0

            global_mask += gaussian_mask 
            class_specific_targets[i] += gaussian_mask

        global_mask = global_mask > 0
        class_specific_targets = class_specific_targets > 0
        
        # # Visualize the results
        # plt.imshow(global_mask, cmap="gray")
        # plt.savefig(f"{results['img_id']}-global_mask_gs.png")
        # plt.close()
        # fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        # for i in range(self.cls_num):
        #     row, col = divmod(i, 5)
        #     axes[row, col].imshow(class_specific_targets[i], cmap="gray")
        #     axes[row, col].set_title(f"Class {i}")

        # plt.tight_layout()
        # plt.savefig(f"{results['img_id']}-class_specific_targets_gs.png")
        # plt.close()

        results['rbox2segmask'] = global_mask.float(), class_specific_targets.float()
        return results

    def __repr__(self):
        return self.__class__.__name__


def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(
        R.permute(0, 2, 1)).reshape(_shape[:-1] + (2, 2))
    sigma_chol = torch.linalg.cholesky(sigma, out=None, upper=False)
    sigma = torch.matmul(sigma_chol, sigma_chol.transpose(-1, -2))

    return xy, sigma


def rbox2GS_mask(xywhr, h, w):
    # input batch of oriented bounding boxes
    # output gaussian mask
    center, cov = xy_wh_r_2_xy_sigma(xywhr)
    # 定义高斯分布，输入的均值和协方差矩阵需要满足batch形式的要求
    gaussian = MultivariateNormal(center, cov)

    # 生成mask，需要使用meshgrid函数生成二维坐标网格，并将其扩展到与中心点和协方差矩阵的batch size相同
    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    pos = (
        torch.stack((x, y), 2)[:, :, None, :]
        .repeat(1, 1, xywhr.shape[0], 1)
        .to(xywhr.device)
    )
    mask = gaussian.log_prob(pos).exp().permute(2, 0, 1)

    # 对每个box的高斯分布掩码进行归一化处理，使得积分为1
    mask_sum = torch.sum(mask, dim=(1, 2), keepdim=True)
    mask_normalized = mask / mask_sum
    
    # max_value = torch.max(mask)
    # mask = mask / max_value
    return mask_normalized


def polygons_to_bitmask_cpu(polygons, height, width) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(np.bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(np.bool_)


def obb2poly_le90(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2, N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()