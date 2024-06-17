# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList

from mmrotate.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage import SingleStageDetector
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


@MODELS.register_module()
class FCOS_TCM(SingleStageDetector):

    def __init__(self,
                 backbone: ConfigType,
                 text_encoder: ConfigType,
                 prompt_generator: ConfigType,
                 vis_context_decoder: ConfigType,
                 text_context_decoder: ConfigType,
                 context_length = 25,
                 tau=0.07,
                 seg_loss=False,
                 seg_loss_weight=1,
                 score_map_idx = [3],  # resnet: 3, vit: 2
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # clip modules
        self.text_encoder = MODELS.build(text_encoder)
        self.prompt_generator = MODELS.build(prompt_generator)
        self.vis_context_decoder = MODELS.build(vis_context_decoder)
        self.text_context_decoder = MODELS.build(text_context_decoder)
        self.context_length = context_length
        self.tau = tau
        self.use_seg_loss = seg_loss
        self.seg_loss_weight = seg_loss_weight
        self.score_map_idx = score_map_idx

        self.class_names = [
        'plane', 'baseball diamond', 'bridge', 'ground track field',
        'small vehicle', 'large vehicle', 'ship', 'tennis court',
        'basketball court', 'storage tank', 'soccer ball field', 'roundabout',
        'harbor', 'swimming pool', 'helicopter'
        ]
        
        if context_length > 10:
            self.texts = torch.cat(
                [clip.tokenize(f"In this aerial photo, a {label} can be found.",
                               context_length=self.context_length) for label in self.class_names])
        else:
            self.texts = torch.cat(
                [clip.tokenize(label, context_length=self.context_length) for label in self.class_names])

        # learnable textual context
        learnable_context = self.text_encoder.context_length - self.context_length
        token_embed_dim = self.text_encoder.transformer_width
        text_dim = self.text_encoder.embed_dim

        self.contexts = nn.Parameter(torch.randn(1, learnable_context, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        self.vis_gamma = nn.Parameter(torch.ones([]) * 1e-3)  # for updating visual embedding diff

    def extract_feat(self, batch_inputs: Tensor, use_seg_loss=False) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        score_maps, text_features = self.compute_text_features(x)
        # score_maps = self.compute_score_maps(x, text_features)
        x = list(x[:-1])
        for i in self.score_map_idx:
            if x[i].shape[2:] != score_maps.shape[-2:]:
                resized_mask = F.interpolate(score_maps, size=x[i].shape[2:], mode="bilinear", align_corners=True)
                x[i] = torch.cat([x[i], resized_mask], dim=1)
            else:
                x[i] = torch.cat([x[i], score_maps], dim=1)
        # TODO concat with other layers
        if self.with_neck:
            x = self.neck(x)
        
        if use_seg_loss:
            return x, score_maps
        else:
            return x
    
    def compute_text_features(self, x):
        """compute text features to each of x
        Args:
            x ([list]): list of features from the backbone, 
                x[4] is the output of attentionpool2d
        """
        global_feat, visual_embeddings = x[4]
        B, C, H, W = visual_embeddings.shape
        visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # prompt_gen = self.prompt_generator(global_feat)
        prompt_gen = None


        # text embeddings is (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts, prompt_gen).expand(B, -1, -1) # use of text_encoder


        vis_prompt_diff = self.vis_context_decoder(visual_embeddings.reshape(B, C, H * W).permute(0, 2, 1),
                                                           text_embeddings)
        vis_prompt_diff = vis_prompt_diff.permute(0, 2, 1).reshape(B, C, H, W)
        visual_embeddings = visual_embeddings + self.vis_gamma * vis_prompt_diff
        
        text_diff = self.text_context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        text_features = F.normalize(text_embeddings, dim=-1)
        visual_embeddings = F.normalize(visual_embeddings, dim=1)
        score_map_attnpool = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_features) / self.tau
        return score_map_attnpool, text_embeddings
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs, self.use_seg_loss)
        if self.use_seg_loss:
            x, score_map = x
        losses = self.bbox_head.loss(x, batch_data_samples)
        if self.use_seg_loss:
            losses['loss_seg'] = self.compute_seg_loss(score_map, batch_data_samples)
        return losses
    
    def compute_seg_loss(self, score_map, batch_data_samples):
        global_mask = torch.stack([_.metainfo['rbox2segmask'][0] for _ in batch_data_samples]).to(score_map.device)
        class_specific_targets = torch.stack([_.metainfo['rbox2segmask'][1] for _ in batch_data_samples]).to(score_map.device)
        
        mask = global_mask[:, None]
        mask = mask.expand(-1, class_specific_targets.shape[1], -1, -1)
        target = class_specific_targets
        if score_map.shape[2:] != target.shape[2:]:
            score_map_expand = F.interpolate(score_map, target.shape[2:], mode='bilinear')
        else:
            score_map_expand = score_map

        loss = F.binary_cross_entropy(torch.sigmoid(score_map_expand), target, weight=mask, reduction='sum')
        loss = loss / (mask.sum()+1e-8)

        loss *= self.seg_loss_weight
        
        return loss

@MODELS.register_module()
class PromptGenerator(nn.Module):
    def __init__(self,
                 visual_dim=1024,
                 token_embed_dim=512,
                 **kwargs
                 ):
        super(PromptGenerator, self).__init__()

        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU(),
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, token_embed_dim)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        '''
        x: (B, D)
        '''
        x = self.prompt_proj(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)