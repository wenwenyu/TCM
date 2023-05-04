import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.runner import BaseModule, Sequential
from mmcv.runner import force_fp32
# from mmseg.ops import resize
import numpy as np

from mmocr.models.builder import HEADS
from mmocr.models import HeadMixin
from mmocr.models.common.losses.dice_loss import DiceLoss
from mmocr.utils import check_argument


@HEADS.register_module()
class DBFP16Head(HeadMixin, BaseModule):
    """The class for DBNet head. calculate bce loss using logits for fp16

    This was partially adapted from https://github.com/MhLiao/DB
    Add self.fp16_enabled = False, @force_fp32
    Args:
        in_channels (int): The number of input channels of the db head.
        with_bias (bool): Whether add bias in Conv2d layer.
        downsample_ratio (float): The downsample ratio of ground truths.
        loss (dict): Config of loss for dbnet.
        postprocessor (dict): Config of postprocessor for dbnet.
    """

    def __init__(
            self,
            in_channels,
            with_bias=False,
            downsample_ratio=1.0,
            loss=dict(type='DBLoss'),
            postprocessor=dict(type='DBPostprocessor', text_repr_type='quad'),
            init_cfg=[
                dict(type='Kaiming', layer='Conv'),
                dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4)
            ],
            train_cfg=None,
            test_cfg=None,
            **kwargs):
        old_keys = ['text_repr_type', 'decoding_type']
        for key in old_keys:
            if kwargs.get(key, None):
                postprocessor[key] = kwargs.get(key)
                warnings.warn(
                    f'{key} is deprecated, please specify '
                    'it in postprocessor config dict. See '
                    'https://github.com/open-mmlab/mmocr/pull/640'
                    ' for details.', UserWarning)
        BaseModule.__init__(self, init_cfg=init_cfg)
        HeadMixin.__init__(self, loss, postprocessor)

        assert isinstance(in_channels, int)

        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.downsample_ratio = downsample_ratio

        self.binarize = Sequential(
            nn.Conv2d(
                in_channels, in_channels // 4, 3, bias=with_bias, padding=1),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2), nn.Sigmoid()) # TODO replace sigmoid with logits

        self.threshold = self._init_thr(in_channels)

        self.fp16_enabled = False

    def diff_binarize(self, prob_map, thr_map, k):
        return torch.reciprocal(1.0 + torch.exp(-k * (prob_map - thr_map)))

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): Shape (batch_size, hidden_size, h, w).

        Returns:
            Tensor: A tensor of the same shape as input.
        """
        prob_map = self.binarize(inputs)
        thr_map = self.threshold(inputs)
        binary_map = self.diff_binarize(prob_map, thr_map, k=50)
        outputs = torch.cat((prob_map, thr_map, binary_map), dim=1)
        return outputs

    def _init_thr(self, inner_channels, bias=False):
        in_channels = inner_channels
        seq = Sequential(
            nn.Conv2d(
                in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2), nn.Sigmoid())
        return seq

    @force_fp32(apply_to=('score_maps',))
    def get_boundary(self, score_maps, img_metas, rescale):
        """Compute text boundaries via post processing.

        Args:
            score_maps (Tensor): The text score map.
            img_metas (dict): The image meta info.
            rescale (bool): Rescale boundaries to the original image resolution
                if true, and keep the score_maps resolution if false.

        Returns:
            dict: A dict where boundary results are stored in
            ``boundary_result``.
        """

        assert check_argument.is_type_list(img_metas, dict)
        assert isinstance(rescale, bool)

        score_maps = score_maps.squeeze()
        boundaries = self.postprocessor(score_maps)
        if isinstance(boundaries, tuple):
            boundaries, score_map, text_mask = boundaries
        else:
            boundaries, score_map, text_mask = boundaries, None, None
        if rescale:
            boundaries = self.resize_boundary(
                boundaries,
                1.0 / self.downsample_ratio / img_metas[0]['scale_factor'])

        results = dict(
            boundary_result=boundaries,
            filename=img_metas[0]['filename'],
            score_map=score_map,
            text_mask=text_mask)

        return results

    @force_fp32(apply_to=('pred_maps',))
    def loss(self, pred_maps, **kwargs):
        """Compute the loss for scene text detection.

        Args:
            pred_maps (Tensor): The input score maps of shape
                :math:`(NxCxHxW)`.

        Returns:
            dict: The dict for losses.
        """
        losses = self.loss_module(pred_maps, self.downsample_ratio, **kwargs)

        return losses



@HEADS.register_module()
class TextSegHead(HeadMixin, BaseModule):
    """The class for Seg head. calculate bce loss using logits for fp16

    Add self.fp16_enabled = False, @force_fp32
    Args:
        in_channels (int): The number of input channels of the db head.
        with_bias (bool): Whether add bias in Conv2d layer.
        downsample_ratio (float): The downsample ratio of ground truths.
        loss (dict): Config of loss for dbnet.
        postprocessor (dict): Config of postprocessor for dbnet.
    """

    def __init__(
            self,
            in_channels, # 256
            text_embedding_dim,#1024
            with_bias=False,
            downsample_ratio=1.0,
            tau=0.07,
            scale_matching_score_map=True,
            loss=dict(type='TextSegLoss'),
            postprocessor=dict(type='TextSegPostprocessor', text_repr_type='quad'),
            init_cfg=[
                dict(type='Kaiming', layer='Conv'),
                dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4)
            ],
            train_cfg=None,
            test_cfg=None,
            **kwargs):
        old_keys = ['text_repr_type', 'decoding_type']
        for key in old_keys:
            if kwargs.get(key, None):
                postprocessor[key] = kwargs.get(key)
                warnings.warn(
                    f'{key} is deprecated, please specify '
                    'it in postprocessor config dict. See '
                    'https://github.com/open-mmlab/mmocr/pull/640'
                    ' for details.', UserWarning)
        BaseModule.__init__(self, init_cfg=init_cfg)
        HeadMixin.__init__(self, loss, postprocessor)

        assert isinstance(in_channels, int)

        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.downsample_ratio = downsample_ratio
        self.tau=tau
        self.scale_matching_score_map=scale_matching_score_map
        if self.scale_matching_score_map:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.tau))

        self.binarize = Sequential(
            nn.Conv2d(
                in_channels, in_channels // 4, 3, bias=with_bias, padding=1),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels, 2, 2))

        self.fc = nn.Linear(text_embedding_dim, in_channels, bias=with_bias)

        self.sigmoid = nn.Sigmoid()# TODO replace nn.Sigmoid with pixel-matching

        self.fp16_enabled = False

    def forward(self, inputs, text_embeddings=None):
        """
        Args:
            inputs (Tensor): Shape (batch_size, hidden_size, h, w).
            text_embeddings (Tensor): Shape (batch_size, N,  hidden_size). like 1xv conv

        Returns:
            Tensor: A tensor of the same shape as input.
        """
        prob_map = self.binarize(inputs) # (B, in_channels, H, W)
        if text_embeddings is not None:
            text_embeddings = self.fc(text_embeddings) #(B, N, in_channels)
            prob_map = self.pixel_class_matching(prob_map, text_embeddings) # (B, N, H, W)
        # prob_map = self.sigmoid(prob_map)

        if self.scale_matching_score_map:
            logit_scale = self.logit_scale.exp()
            prob_map = logit_scale * prob_map # logits
        else:
            prob_map = prob_map / self.tau # logits
        return prob_map

    def pixel_class_matching(self, visual_embeddings, text_embeddings):
        ''' calculate pixel-class prompt text matching score map
         visual_embeddings: (B, C//4, H, W)
         text_embeddings: (B, N, C//4)
         '''
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2, eps=1e-12)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        return score_map

    @force_fp32(apply_to=('score_maps',))
    def get_boundary(self, score_maps, img_metas, rescale):
        """Compute text boundaries via post processing.

        Args:
            score_maps (Tensor): The text score map.
            img_metas (dict): The image meta info.
            rescale (bool): Rescale boundaries to the original image resolution
                if true, and keep the score_maps resolution if false.

        Returns:
            dict: A dict where boundary results are stored in
            ``boundary_result``.
        """

        assert check_argument.is_type_list(img_metas, dict)
        assert isinstance(rescale, bool)

        # score_maps = self.sigmoid(score_maps) # move to postprocess process
        if score_maps.dim() == 4:
            score_maps = score_maps.squeeze(0) # remove bs dim
        boundaries = self.postprocessor(score_maps)

        if rescale:
            boundaries = self.resize_boundary(
                boundaries,
                1.0 / self.downsample_ratio / img_metas[0]['scale_factor'])

        results = dict(
            boundary_result=boundaries, filename=img_metas[0]['filename'])

        return results

    @force_fp32(apply_to=('pred_maps',))
    def loss(self, pred_maps, **kwargs):
        """Compute the loss for scene text detection.

        Args:
            pred_maps (Tensor): The input score maps of shape
                :math:`(NxCxHxW)`.

        Returns:
            dict: The dict for losses.
        """
        # pred_maps = self.sigmoid(pred_maps)
        losses = self.loss_module(pred_maps, self.downsample_ratio, **kwargs)

        return losses


@HEADS.register_module()
class IdentityHead(BaseModule):
    """The class for DBNet aux seg head.

    Args:
        in_channels (int): The number of input channels of the db head.
        with_bias (bool): Whether add bias in Conv2d layer.
        downsample_ratio (float): The downsample ratio of pred maps.
        loss (dict): Config of loss for dbnet.
        postprocessor (dict): Config of postprocessor for dbnet.
    """

    def __init__(
            self,
            downsample_ratio=1.0,
            loss_weight=1,
            reduction='mean',
            negative_ratio=3.0,
            eps=1e-6,
            bbce_loss=False,
            init_cfg=[
                dict(type='Kaiming', layer='Conv'),
                dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4)
            ],
            **kwargs):
        BaseModule.__init__(self, init_cfg=init_cfg)

        assert reduction in ['mean',
                             'sum'], " reduction must in ['mean','sum']"
        self.downsample_ratio = float(downsample_ratio)
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.bbce_loss = bbce_loss
        self.dice_loss = DiceLoss(eps=eps)

        self.sigmod = nn.Sigmoid()

        self.fp16_enabled = False

    def balance_bce_loss(self, pred, gt, mask):

        positive = (gt * mask)
        negative = ((1 - gt) * mask)
        positive_count = int(positive.float().sum())
        negative_count = min(
            int(negative.float().sum()),
            int(positive_count * self.negative_ratio))

        assert gt.max() <= 1 and gt.min() >= 0
        # assert pred.max() <= 1 and pred.min() >= 0
        # gt: (N, H, W), pred: (N, 1, H, W)
        if len(gt.size()) != pred.size():
            gt = gt.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        # loss = F.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()

        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
                positive_count + negative_count + self.eps)

        return balance_loss

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

    @force_fp32(apply_to=('pred_prob',))
    def loss(self, pred_prob, gt_shrink, gt_shrink_mask, **kwargs):
        assert isinstance(gt_shrink, list)
        assert isinstance(gt_shrink_mask, list)

        # recover pred to origin input image size (gt size)
        pred_size = pred_prob.shape[2:]
        rescale_size = tuple([x * int(self.downsample_ratio) for x in pred_size]) # stage3 1/32, stage2 1/16
        pred_prob = F.interpolate(input=pred_prob, size=rescale_size, mode='bilinear', align_corners=False)
        # pred_prob = F.upsample(pred_prob, rescale_size, mode='bilinear')
        # N, 1, H, W
        feature_sz = pred_prob.size()

        keys = ['gt_shrink', 'gt_shrink_mask']
        gt = {}
        for k in keys:
            gt[k] = eval(k)
            gt[k] = [item.rescale(1.0) for item in gt[k]] # cpu operation, time-consuming if rescale isn't 1.0
            gt[k] = self.bitmasks2tensor(gt[k], feature_sz[2:])
            gt[k] = [item.to(pred_prob.device) for item in gt[k]]
        gt['gt_shrink'][0] = (gt['gt_shrink'][0] > 0).float()

        loss_prob = self._loss(pred_prob, gt['gt_shrink'][0], gt['gt_shrink_mask'][0])

        results = dict(
            loss_pix_cls=self.loss_weight * loss_prob)

        return results

    def _loss(self, pred_prob, gt, gt_mask):
        if self.bbce_loss:
            loss_prob = self.balance_bce_loss(pred_prob, gt, gt_mask)
        else:
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1, https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/dice.py
            pred_prob = F.logsigmoid(pred_prob).exp()  # binary_class
            # pred_prob = pred_prob.log_softmax(dim=1).exp() # multiclass
            loss_prob = self.dice_loss(pred_prob, gt, gt_mask)
        return loss_prob

    def forward(self, inputs):
        # inputs matching score mpa has been normalized and value is [0, 1],
        # but it divide by tau, so here inputs is logits
        return inputs
        # return self.sigmod(inputs)


@HEADS.register_module()
class FCEIdentityHead(IdentityHead):
    """The class for DBNet aux seg head.

    Args:
        in_channels (int): The number of input channels of the db head.
        with_bias (bool): Whether add bias in Conv2d layer.
        downsample_ratio (float): The downsample ratio of ground truths.
        loss (dict): Config of loss for dbnet.
        postprocessor (dict): Config of postprocessor for dbnet.
    """
    # def __init__(self, pix_text_match_stage_index, **kwargs):
    #     '''
    #     pix_text_match_stage_index: 0: 1/8, 1: 1/16, 2: 1/32
    #     '''
    #     self.pix_text_match_stage_index = pix_text_match_stage_index
    #     IdentityHead.__init__(**kwargs)

    @force_fp32(apply_to=('pred_prob',))
    def loss(self, pred_prob, matching_maps):
        """Compute FCENet pix-text matching loss.

        Args:
            pred_prob : pix-text matching score map
            matching_maps (list[ndarray]): List of pix-text matching ground truth target map
                with shape :math:`(C, H, W)`. stage 0: 1/8, 1: 1/16, 2: 1/32

        Returns:
            dict:  A loss dict
        """
        # assert isinstance(pred_prob, list)
        # assert p3_maps[0].shape[0] == 4 * self.fourier_degree + 5, \
        #     'fourier degree not equal in FCEhead and FCEtarget'

        device = pred_prob.device
        # to tensor
        gt = torch.from_numpy(np.stack(matching_maps)).float().to(device)
        # gt = gt.permute(0, 2, 3, 1).contiguous() # N, H, W, C
        tr_mask = gt[:, 0, :, :]
        # tcl_mask = gt[:, 1, :]
        train_mask = gt[:, 2, :, :]

        # tr_train_mask = train_mask * tr_mask

        loss_prob = self._loss(pred_prob, tr_mask.float(), train_mask.long())

        # loss_pix_cls = self.loss_weight * loss_prob

        results = dict(
            loss_pix_cls=self.loss_weight * loss_prob)

        return results


@HEADS.register_module()
class PANIdentityHead(IdentityHead):
    """The class for DBNet aux seg head.

    Args:
        in_channels (int): The number of input channels of the db head.
        with_bias (bool): Whether add bias in Conv2d layer.
        downsample_ratio (float): The downsample ratio of ground truths.
        loss (dict): Config of loss for dbnet.
        postprocessor (dict): Config of postprocessor for dbnet.
    """
    # def __init__(self, pix_text_match_stage_index, **kwargs):
    #     '''
    #     pix_text_match_stage_index: 0: 1/8, 1: 1/16, 2: 1/32
    #     '''
    #     self.pix_text_match_stage_index = pix_text_match_stage_index
    #     IdentityHead.__init__(**kwargs)

    @force_fp32(apply_to=('pred_prob',))
    def loss(self, pred_prob, gt_kernels, gt_mask):
        """Compute FCENet pix-text matching loss.

        Args:
            pred_prob : pix-text matching score map
            gt_kernels (list[BitmapMasks]): The kernel list with each element
                being the text kernel mask for one img.
            gt_mask (list[BitmapMasks]): The effective mask list
                with each element being the effective mask for one img.
        Returns:
            dict:  A loss dict
        """
        # assert isinstance(pred_prob, list)
        # assert p3_maps[0].shape[0] == 4 * self.fourier_degree + 5, \
        #     'fourier degree not equal in FCEhead and FCEtarget'

        pred_size = pred_prob.shape[2:]
        rescale_size = tuple([x * int(1/self.downsample_ratio) for x in pred_size]) # stage3 1/32, stage2 1/16
        pred_prob = F.interpolate(input=pred_prob, size=rescale_size, mode='bilinear', align_corners=False)
        # pred_prob = F.upsample(pred_prob, rescale_size, mode='bilinear')
        # N, 1, H, W
        feature_sz = pred_prob.size()

        mapping = {'gt_kernels': gt_kernels, 'gt_mask': gt_mask}
        gt = {}
        for key, value in mapping.items():
            gt[key] = value
            # gt[key] = [item.rescale(self.downsample_ratio) for item in gt[key]] # time consuming, so resize pred instead of gt
            gt[key] = [item.rescale(1.0) for item in gt[key]]
            gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
            gt[key] = [item.to(pred_prob.device) for item in gt[key]]


        text_gt =  gt['gt_kernels'][0]
        text_gt_mask = gt['gt_mask'][0]
        pred_prob = pred_prob.squeeze(1)
        text_gt = text_gt.squeeze(1)
        text_gt_mask = text_gt_mask.squeeze(1)
        # compute text loss
        sampled_mask = self.ohem_batch(pred_prob.detach(),
                                       text_gt, text_gt_mask)
        loss_prob = self.dice_loss_with_logits(pred_prob,
                                               text_gt,
                                                sampled_mask)

        if self.reduction == 'mean':
            loss_prob = loss_prob.mean()
        elif self.reduction == 'sum':
            loss_prob = loss_prob.sum()
        else:
            raise NotImplementedError

        results = dict(
            loss_pix_cls=self.loss_weight * loss_prob)

        return results


    def ohem_img(self, text_score, gt_text, gt_mask):
        """Sample the top-k maximal negative samples and all positive samples.

        Args:
            text_score (Tensor): The text score of size :math:`(H, W)`.
            gt_text (Tensor): The ground truth text mask of size
                :math:`(H, W)`.
            gt_mask (Tensor): The effective region mask of size :math:`(H, W)`.

        Returns:
            Tensor: The sampled pixel mask of size :math:`(H, W)`.
        """
        assert isinstance(text_score, torch.Tensor)
        assert isinstance(gt_text, torch.Tensor)
        assert isinstance(gt_mask, torch.Tensor)
        assert len(text_score.shape) == 2
        assert text_score.shape == gt_text.shape
        assert gt_text.shape == gt_mask.shape

        pos_num = (int)(torch.sum(gt_text > 0.5).item()) - (int)(
            torch.sum((gt_text > 0.5) * (gt_mask <= 0.5)).item())
        neg_num = (int)(torch.sum(gt_text <= 0.5).item())
        neg_num = (int)(min(pos_num * self.negative_ratio, neg_num))

        if pos_num == 0 or neg_num == 0:
            warnings.warn('pos_num = 0 or neg_num = 0')
            return gt_mask.bool()

        neg_score = text_score[gt_text <= 0.5]
        neg_score_sorted, _ = torch.sort(neg_score, descending=True)
        threshold = neg_score_sorted[neg_num - 1]
        sampled_mask = (((text_score >= threshold) + (gt_text > 0.5)) > 0) * (
                gt_mask > 0.5)
        return sampled_mask

    def ohem_batch(self, text_scores, gt_texts, gt_mask):
        """OHEM sampling for a batch of imgs.

        Args:
            text_scores (Tensor): The text scores of size :math:`(H, W)`.
            gt_texts (Tensor): The gt text masks of size :math:`(H, W)`.
            gt_mask (Tensor): The gt effective mask of size :math:`(H, W)`.

        Returns:
            Tensor: The sampled mask of size :math:`(H, W)`.
        """
        assert isinstance(text_scores, torch.Tensor)
        assert isinstance(gt_texts, torch.Tensor)
        assert isinstance(gt_mask, torch.Tensor)
        assert len(text_scores.shape) == 3
        assert text_scores.shape == gt_texts.shape
        assert gt_texts.shape == gt_mask.shape

        sampled_masks = []
        for i in range(text_scores.shape[0]):
            sampled_masks.append(
                self.ohem_img(text_scores[i], gt_texts[i], gt_mask[i]))

        sampled_masks = torch.stack(sampled_masks)

        return sampled_masks

    def dice_loss_with_logits(self, pred, target, mask):

        smooth = 0.001

        pred = torch.sigmoid(pred)
        target[target <= 0.5] = 0
        target[target > 0.5] = 1
        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        pred = pred * mask
        target = target * mask

        a = torch.sum(pred * target, 1) + smooth
        b = torch.sum(pred * pred, 1) + smooth
        c = torch.sum(target * target, 1) + smooth
        d = (2 * a) / (b + c)
        return 1 - d