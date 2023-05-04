# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 12/25/21 3:52 PM

import torch
import torch.nn.functional as F
from torch import nn

from mmocr.models.builder import LOSSES
from mmocr.models.common.losses.dice_loss import DiceLoss


@LOSSES.register_module()
class TextSegLoss(nn.Module):
    """The class for implementing text seg loss. modified alpha to loss_db

    This is partially adapted from https://github.com/MhLiao/DB.

    Args:
        loss_weight (float): The binary loss coef.
        reduction (str): The way to reduce the loss.
        negative_ratio (float): The ratio of positives to negatives.
        eps (float): Epsilon in the threshold loss function.
        bbce_loss (bool): Whether to use balanced bce for probability loss.
            If False, dice loss will be used instead.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 negative_ratio=3.0,
                 with_logits=True,
                 eps=1e-6,
                 bbce_loss=False):
        super().__init__()
        assert reduction in ['mean',
                             'sum'], " reduction must in ['mean','sum']"
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.negative_ratio = negative_ratio
        self.with_logits=with_logits
        self.eps = eps
        self.bbce_loss = bbce_loss
        self.dice_loss = DiceLoss(eps=eps)

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
        """
        assert isinstance(bitmasks, list)
        assert isinstance(target_sz, tuple)

        batch_size = len(bitmasks)
        num_levels = len(bitmasks[0])

        result_tensors = []

        for level_inx in range(num_levels):
            kernel = []
            for batch_inx in range(batch_size):
                mask = torch.from_numpy(bitmasks[batch_inx].masks[level_inx])
                mask_sz = mask.shape
                pad = [
                    0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]
                ]
                mask = F.pad(mask, pad, mode='constant', value=0)
                kernel.append(mask)
            kernel = torch.stack(kernel)
            result_tensors.append(kernel)

        return result_tensors

    def balance_bce_loss(self, pred, gt, mask):

        positive = (gt * mask)
        negative = ((1 - gt) * mask)
        positive_count = int(positive.float().sum())
        negative_count = min(
            int(negative.float().sum()),
            int(positive_count * self.negative_ratio))

        assert gt.max() <= 1 and gt.min() >= 0
        # assert pred.max() <= 1 and pred.min() >= 0
        if self.with_logits:
            loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        else:
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()

        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
                positive_count + negative_count + self.eps)

        return balance_loss

    def forward(self, preds, downsample_ratio, gt_shrink, gt_shrink_mask,
                gt_thr, gt_thr_mask):
        """Compute DBNet loss.

        Args:
            preds (Tensor): The output tensor with size :math:`(N, 1, H, W)`, logits.
            downsample_ratio (float): The downsample ratio for the
                ground truths.
            gt_shrink (list[BitmapMasks]): The mask list with each element
                being the shrunk text mask for one img.
            gt_shrink_mask (list[BitmapMasks]): The effective mask list with
                each element being the shrunk effective mask for one img.
            gt_thr (list[BitmapMasks]): The mask list with each element
                being the threshold text mask for one img.
            gt_thr_mask (list[BitmapMasks]): The effective mask list with
                each element being the threshold effective mask for one img.

        Returns:
            dict: The dict for dbnet losses with "loss_prob", "loss_db" and
            "loss_thresh".
        """
        assert isinstance(downsample_ratio, float)

        assert isinstance(gt_shrink, list)
        assert isinstance(gt_shrink_mask, list)
        assert isinstance(gt_thr, list)
        assert isinstance(gt_thr_mask, list)

        pred_prob = preds[:, 0, :, :] # logits
        feature_sz = preds.size()

        keys = ['gt_shrink', 'gt_shrink_mask']
        gt = {}
        for k in keys:
            gt[k] = eval(k)
            gt[k] = [item.rescale(downsample_ratio) for item in gt[k]]
            gt[k] = self.bitmasks2tensor(gt[k], feature_sz[2:])
            gt[k] = [item.to(preds.device) for item in gt[k]]
        gt['gt_shrink'][0] = (gt['gt_shrink'][0] > 0).float()
        if self.bbce_loss:
            loss_prob = self.balance_bce_loss(pred_prob, gt['gt_shrink'][0],
                                              gt['gt_shrink_mask'][0])
        else:
            if self.with_logits:
                pred_prob = F.logsigmoid(pred_prob).exp() # binary_class
            # pred_prob = pred_prob.log_softmax(dim=1).exp() # multiclass
            loss_prob = self.dice_loss(pred_prob, gt['gt_shrink'][0],
                                       gt['gt_shrink_mask'][0])

        results = dict(loss_prob=self.loss_weight * loss_prob)

        return results
