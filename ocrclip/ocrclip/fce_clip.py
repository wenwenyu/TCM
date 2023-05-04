# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 12/25/21 10:05 AM

import warnings

import numpy as np
import cv2
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.detectors import \
    SingleStageDetector as MMDET_SingleStageDetector
from mmocr.models.builder import (DETECTORS, build_backbone, build_head,
                                  build_neck)
import mmcv
from mmocr.core import imshow_pred_boundary, show_pred_gt
from mmocr.core.visualize import tile_image
from mmcv.utils import check_file_exist, is_str, mkdir_or_exist

from mmocr.models import SingleStageTextDetector, TextDetectorMixin
from mmseg.core import add_prefix
from mmseg.ops import resize
import matplotlib.pyplot as plt

from .untils import tokenize
from .ocrclip import OCRCLIP

@DETECTORS.register_module()
class FCECLIP(OCRCLIP):
    """The class for implementing DBNet text detector: Real-time Scene Text
    Detection with Differentiable Binarization.

    Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.

    DBCLIP
    [https://arxiv.org/abs/1911.08947].
    """

    def _bbox_head_forward_train(self, x, text_embeddings, **kwargs):
        '''
        kwargs ：
        '''
        losses = dict()
        preds = self.bbox_head(x) # (B, 3, H, W)
        #
        newkwargs = kwargs.copy()
        matching_maps = newkwargs.pop('matching_maps')
        loss_bbox = self.bbox_head.loss(preds, **newkwargs)
        losses.update(add_prefix(loss_bbox, 'bbox'))
        return losses


    def _identity_head_forward_train(self, x, img_metas, **kwargs):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        preds = self.identity_head(x)
        matching_maps = kwargs.pop('matching_maps')
        loss_bbox = self.identity_head.loss(preds, matching_maps = matching_maps)
        losses.update(add_prefix(loss_bbox, 'aux'))
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        _x_orig = [x[i] for i in range(len(x))]

        # BKC, BKHW
        text_embeddings, score_map = self.instance_class_matching(x)

        x_orig = list(x[0:-1])
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)

        if self.with_neck:
            x_fusion = self.neck(x_orig)
        else:
            x_fusion = x_orig[0]
            # score_map_up = resize(
            #     input=score_map,
            #     size=x_fusion.shape[2:],
            #     mode='bilinear',
            #     align_corners=False)
            # x_fusion = torch.cat([x_fusion, score_map_up], dim=1)

        # if self.text_head:
        #     x = [text_embeddings, ] + x_fusion
        # else:
        x = x_fusion

        outs = self.bbox_head(x)

        boundaries = self.bbox_head.get_boundary(outs, img_metas, rescale)

        return [boundaries]



