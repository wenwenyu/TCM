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
from .feature_visualization import draw_feature_map

@DETECTORS.register_module()
class OCRCLIP(TextDetectorMixin, MMDET_SingleStageDetector):
    """The class for implementing DBNet text detector: Real-time Scene Text
    Detection with Differentiable Binarization.

    Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.

    DBCLIP
    [https://arxiv.org/abs/1911.08947].
    """

    def __init__(self,
                 backbone,
                 text_encoder,
                 context_decoder, # for text embedding query updated by CA(text, visual)
                 bbox_head,
                 class_names,
                 context_length, # len of predefine text
                 seq_context_length=5,
                 context_feature='attention',
                 use_learnable_prompt=True,  # predefine text + learnable prompt
                 use_learnable_prompt_only=False, # only use learnable prompt
                 use_context_decoder=True, # for updating text embedding by ca
                 prompt_generator=None, # for text prompt generator by linear(global_feat.)
                 score_concat_index=3, # 3:HW/32, 2:HW/16
                 text_head=False,
                 visual_prompt_generator=None, # visual prompt generator
                 neck=None,
                 tau=0.07,
                 scale_matching_score_map=False,
                 auxiliary_head=None,
                 identity_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 show_score=False,
                 init_cfg=None,
                 token_embed_dim=512, text_dim=1024,
                 is_v2=False,
                 **args):
        super(MMDET_SingleStageDetector, self).__init__(init_cfg=init_cfg)
        TextDetectorMixin.__init__(self, show_score)

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')

            assert backbone.get('pretrained') is None, \
                'both backbone and detector set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and detector set pretrained weight'

            if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained:
                print('not CLIP pre-trained weight, using CLIP ViT-B-16')
                text_encoder.pretrained = 'pretrained/ViT-B-16.pt'
            else:
                text_encoder.pretrained = pretrained

        self.backbone = build_backbone(backbone)

        self.text_encoder = build_backbone(text_encoder)
        self.context_decoder = build_backbone(context_decoder)
        if prompt_generator is not None:
            self.prompt_generator = build_backbone(prompt_generator)
        else:
            self.prompt_generator = None
        if visual_prompt_generator is not None:
            self.visual_prompt_generator = build_backbone(visual_prompt_generator)
        else:
            self.visual_prompt_generator = None

        self.context_length = context_length
        self.seq_context_length = seq_context_length
        self.score_concat_index = score_concat_index

        assert context_feature in ['attention', 'backbone']
        self.context_feature = context_feature

        self.text_head = text_head
        self.tau = tau
        self.scale_matching_score_map=scale_matching_score_map
        self.use_learnable_prompt = use_learnable_prompt
        self.use_learnable_prompt_only = use_learnable_prompt_only
        if use_learnable_prompt_only:
            assert use_learnable_prompt == use_learnable_prompt_only
        self.use_context_decoder = use_context_decoder
        if not use_learnable_prompt: # original clip text encoder
            self.context_length = self.text_encoder.context_length

        if neck is not None:
            self.neck = build_neck(neck)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.with_auxiliary_head = False
        self.auxiliary_head = None
        self._init_auxiliary_head(auxiliary_head)

        self.with_identity_head = False
        self.identity_head = None
        self._init_identity_head(identity_head)

        # input of text encoder of clip
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(self.texts)

        if not self.use_learnable_prompt_only:
            learnable_context_length = self.text_encoder.context_length - self.context_length
        else:
            learnable_context_length = self.text_encoder.context_length # totally use learnable prompt
        # text encoder's learnable prompt template for class name （text encoder最大支持长度-class表示长度, 13-5）
        self.contexts = nn.Parameter(torch.randn(1, learnable_context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)

        # assert self.text_encoder.context_length >= self.seq_context_length, 'text_encoder.context_length must larger than seq_context_length'
        # seq_prompt_len = self.text_encoder.context_length - self.seq_context_length
        # self.seq_contexts = nn.Parameter(torch.randn(1, seq_prompt_len, token_embed_dim))
        # nn.init.trunc_normal_(self.seq_contexts)
        if self.scale_matching_score_map: # origin clip behavior, instead of divide by self.tau on pix-cls matching score map
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.tau))

        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-3) # for updating text embedding diff
        self.vis_gamma = nn.Parameter(torch.ones([]) * 1e-3)  # for updating visual embedding diff

        self.visualization_feat = False # for visualization only

        self.is_v2 = is_v2
        if self.is_v2:
            self.meta_query = nn.Parameter(torch.randn(1024))
            self.pix_w = nn.Parameter(torch.ones(1024, 512))
        assert self.with_bbox

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary image-sequence matching head``"""
        if auxiliary_head is not None:
            self.with_auxiliary_head = True
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(build_head(head_cfg))
            else:
                self.auxiliary_head = build_head(auxiliary_head)

    def _init_identity_head(self, identity_head):
        """Initialize ``auxiliary pixel-text matching head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = build_head(identity_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        return x

    def _bbox_head_forward_train(self, x, text_embeddings, **kwargs):
        '''
        kwargs ： db gt_shrink, gt_shrink_mask, gt_thr, gt_thr_mask
        '''
        losses = dict()
        if self.bbox_head.__class__.__name__ == 'TextSegHead':
            preds = self.bbox_head(x, text_embeddings) # (B, 1, H, W)
        else:
            preds = self.bbox_head(x) # (B, 3, H, W)
        # gt_shrink, gt_shrink_mask, gt_thr, gt_thr_mask
        loss_bbox = self.bbox_head.loss(preds, **kwargs)
        losses.update(add_prefix(loss_bbox, 'bbox'))
        return losses

    def _auxiliary_head_forward_train(self, x, img_metas):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        preds = self.auxiliary_head(x)
        loss_bbox = self.auxiliary_head.loss(preds, img_metas=img_metas)
        losses.update(add_prefix(loss_bbox, 'aux'))
        return losses

    def _identity_head_forward_train(self, x, img_metas, **kwargs):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        preds = self.identity_head(x)
        loss_bbox = self.identity_head.loss(preds, **kwargs)
        losses.update(add_prefix(loss_bbox, 'aux'))
        return losses

    def after_extract_feat(self, x):
        ''' calculate pixel-class matching score map '''
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            # (B, C, 1+H*W)
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
                                       dim=2).permute(0, 2, 1)  # B, N, C

        # texts is set of classes name embedding, contexts is learnable prompt embedding
        # (B, K, C), 取text_embeddings最后时刻t的值作为输出(BKLC->BKC)
        contexts = self.contexts if self.use_learnable_prompt else None
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), contexts).expand(B, -1, -1)
        if self.use_context_decoder:
            # update text_embeddings by visual_context, post-model prompting refines the text_embeddings
            text_diff = self.context_decoder(text_embeddings, visual_context)
            # (B, K, C)
            text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map

    def instance_class_matching(self, x, img_name=None):
        ''' calculate instance-class prompt text matching score map '''

        global_feat, visual_embeddings = x[-1]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            # (B, C, 1+H*W)
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
                                       dim=2).permute(0, 2, 1)  # B, N, C

        # texts is set of classes name embedding, contexts is learnable prompt embedding
        prompt_gen = None
        # text prompting
        if self.prompt_generator is not None: # text prompt generator
            if self.is_v2:
                prompt_gen = self.prompt_generator(self.meta_query.expand(B, -1)) # (B, C)
            else:
                prompt_gen = self.prompt_generator(global_feat) # (B, C)

        contexts = self.contexts if self.use_learnable_prompt else None # (1, N, C)
        # (B, K, C), last time step t as output, (BKLC->BKC)
        # (1, K, D) -> (B, K, D)
        text_embeddings = self.text_encoder(
                                self.texts.to(global_feat.device),
                                contexts,
                                use_learnable_prompt_only=self.use_learnable_prompt_only,
                                prompt_gen=prompt_gen).expand(B, -1, -1)
        
        if self.is_v2:
            global_feat_ = global_feat@self.pix_w
            global_feat_ = F.normalize(global_feat_, dim=-1, p=2).unsqueeze(1) # (B,1,D)
            text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
            cossim = torch.sum(text_embeddings * global_feat_, dim=-1).unsqueeze(-1) # (B,K,1)
            text_embeddings = cossim * global_feat_ + text_embeddings
        
        # visual prompting
        # import pdb;pdb.set_trace()
        if self.visual_prompt_generator is not None:
            vis_prompt_diff = self.visual_prompt_generator(visual_embeddings.reshape(B, C, H * W).permute(0, 2, 1),
                                                           text_embeddings)
            vis_prompt_diff = vis_prompt_diff.permute(0, 2, 1).reshape(B, C, H, W)

            if self.visualization_feat:
                draw_feature_map(vis_prompt_diff, title='vis_prompt', img_name=img_name)

            visual_embeddings = visual_embeddings + self.vis_gamma * vis_prompt_diff

            # if self.visualization_feat:
            #     draw_feature_map(visual_embeddings, title='added_vis_pro_img_embed', img_name=img_name)

            # if self.visualization_feat:
            #     draw_feature_map(visual_embeddings, title='vis_prompt', img_name=img_name)

        if self.use_context_decoder:
            # update text_embeddings by visual_context, post-model prompting refines the text_embeddings
            text_diff = self.context_decoder(text_embeddings, visual_context)
            # (B, K, C)
            text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        if self.scale_matching_score_map:
            logit_scale = self.logit_scale.exp()
            score_map = logit_scale * score_map # logits
        return text_embeddings, score_map

    def image_seq_matching(self, x, seq):
        ''' calculate image-sequence prompt matching score map
            seq: B, L, read-order text sequence of image
        '''
        # (B, C), (B, C, H, W)
        image_features, visual_embeddings = x[-1]

        # (B, seq_ctx_len)
        seq_id = torch.cat([tokenize('this is '+s, context_length=self.seq_context_length, truncate=True) for s in seq])
        # (B, C)
        seq_features = self.text_encoder(seq_id.to(image_features.device), None)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        seq_features = seq_features / seq_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        img_seq_logit_scale = self.img_seq_logit_scale.exp()
        logits_per_image = img_seq_logit_scale * image_features @ seq_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.

                kwargs: gt_shrink, gt_shrink_mask, gt_thr, gt_thr_mask
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        # 4 stage output + (global feat, self-attention based feat)
        x = self.extract_feat(img)
        _x_orig = [x[i] for i in range(len(x)-1)]

        # BKC, BKHW
        text_embeddings, score_map = self.instance_class_matching(x)

        x_orig = list(x[0:-1])
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)

        if self.with_neck:
            x_fusion = self.neck(x_orig)  # (N, C, H/4, W/4)
            # _x_orig = x_fusion  # 此次是否需要放if判断外面
        else:
            x_fusion = x_orig[0]
            # score_map_up = resize(
            #     input=score_map,
            #     size=x_fusion.shape[2:],
            #     mode='bilinear',
            #     align_corners=False)
            # x_fusion = torch.cat([x_fusion, score_map_up], dim=1)

        losses = dict()
        # if self.text_head:
        #     x = [text_embeddings, ] + x_fusion
        # else:
        x = x_fusion

        loss_bbox = self._bbox_head_forward_train(x, text_embeddings, **kwargs)
        losses.update(loss_bbox)

        if self.with_identity_head:
            loss_identity = self._identity_head_forward_train(
                score_map if self.scale_matching_score_map else score_map / self.tau,
                img_metas, **kwargs)
            losses.update(loss_identity)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas)
            losses.update(loss_aux)

        return losses
    def forward_dummy(self, img):
        # used for getting flops
        x = self.extract_feat(img)
        _x_orig = [x[i] for i in range(len(x)-1)]

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
        if self.bbox_head.__class__.__name__ == 'TextSegHead':
            outs = self.bbox_head(x, text_embeddings)
        else:
            outs = self.bbox_head(x)
        # logits
        return outs

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat(img)
        _x_orig = [x[i] for i in range(len(x)-1)]

        if self.visualization_feat:
            # for visualization
            ori_filename = img_metas[0]['ori_filename']
            ori_filename = osp.basename(ori_filename)
            draw_feature_map(x[self.score_concat_index], title='img_embed', img_name=ori_filename)

        # BKC, BKHW
        if self.visualization_feat:
            text_embeddings, score_map = self.instance_class_matching(x, img_name=ori_filename)
        else:
            text_embeddings, score_map = self.instance_class_matching(x)

        # if self.visualization_feat:
        #     draw_feature_map(score_map, title='scoremap')

        # new_scoremap = score_map if self.scale_matching_score_map else score_map / self.tau
        # if self.visualization_feat:
        #     draw_feature_map(score_map/self.tau, title='scoremap_tau')

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
        if self.bbox_head.__class__.__name__ == 'TextSegHead':
            outs = self.bbox_head(x, text_embeddings)
        else:
            outs = self.bbox_head(x)

        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return outs

        if len(img_metas) > 1:
            boundaries = [
                self.bbox_head.get_boundary(*(outs[i].unsqueeze(0)),
                                            [img_metas[i]], rescale)
                for i in range(len(img_metas))
            ]

        else:
            boundaries = [
                self.bbox_head.get_boundary(*outs, img_metas, rescale)
            ]

        return boundaries


    def show_result_with_score_map(self,
                    img,
                    result,
                    gt_masks=None,
                    img_meta=None,
                    score_thr=0.5,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results to draw over `img`.
                            keys: boundary_result, filename, score_map, text_mask
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.imshow_pred_boundary`
        """
        img = mmcv.imread(img)
        img_ori = img.copy()
        img = img.copy()
        boundaries = None
        labels = None
        if 'boundary_result' in result.keys():
            boundaries = result['boundary_result']
            labels = [0] * len(boundaries)
        score_map = result['score_map']
        # score_map = (result['score_map']*255).astype(np.uint8)
        text_mask = result['text_mask']
        # text_mask = (result['text_mask']*255).astype(np.uint8)

        h, w, _ = img_meta['img_shape']
        score_map = score_map[:h, :w] # remove padding
        text_mask = text_mask[:h, :w] # remove padding

        ori_h, ori_w = img_meta['ori_shape'][:-1] # recover to ori_sz
        score_map = mmcv.imresize(score_map, (ori_w, ori_h))
        text_mask = mmcv.imresize(text_mask, (ori_w, ori_h))

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        if boundaries is not None and len(boundaries) != 0:
            img_with_pred = imshow_pred_boundary(
                img,
                boundaries,
                labels,
                score_thr=score_thr,
                boundary_color=bbox_color,
                text_color=text_color,
                thickness=thickness,
                font_scale=font_scale,
                win_name=win_name,
                show=False,
                wait_time=wait_time,
                out_file=None, #out_file,
                show_score=True)
        else:
            img_with_pred = img

        tile_img_name = out_file.split('.')[0] + '_tile.' + out_file.split('.')[-1]
        dir_name = osp.abspath(osp.dirname(tile_img_name))
        mkdir_or_exist(dir_name)

        # score_map_img_name = out_file.split('.')[0] + '_score.' + out_file.split('.')[-1]
        # text_mask_img_name = out_file.split('.')[0] + '_mask.' + out_file.split('.')[-1]
        # print(f'save {tile_img_name}')

        if gt_masks is not None:
            cv2.polylines(
                img_with_pred, [np.array(gt_masks_ins).astype(np.int32).reshape(-1, 1, 2) for gt_masks_ins in gt_masks],
                True,
                color=(0, 0, 255), # bgr
                thickness=1)
        # if 'ic15' in tile_img_name:
        plt.figure(figsize=(48,12))
        # else:
            # plt.figure(figsize=(18,6))
        # plt.plot(range(10))
        plt.subplot(141)
        plt.imshow(score_map, cmap='jet');plt.axis('off');plt.title('score')#;plt.colorbar()
        plt.subplot(142)
        plt.imshow(text_mask, cmap='gray');plt.axis('off');plt.title('mask')
        plt.subplot(143)
        img_with_pred = mmcv.bgr2rgb(img_with_pred)
        plt.imshow(img_with_pred);plt.axis('off');plt.title('pred_GT')
        plt.subplot(144)
        img_ori = mmcv.bgr2rgb(img_ori)
        plt.imshow(img_ori);plt.axis('off');plt.title('img_ori')
        plt.savefig(tile_img_name)

        # mmcv.imwrite(tile_img, tile_img_name)
        # mmcv.imwrite(score_map, score_map_img_name)
        # mmcv.imwrite(text_mask, text_mask_img_name)


        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, '
                          'result image will be returned')
        return img_with_pred


