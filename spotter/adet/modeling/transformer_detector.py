from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess as d2_postprocesss
from detectron2.structures import ImageList, Instances
from detectron2.layers import cat

from adet.layers.pos_encoding import PositionalEncoding2D
from adet.modeling.testr.losses import SetCriterion, DiceLoss
from adet.modeling.testr.matcher import build_matcher
from adet.modeling.testr.models import TESTR
from adet.utils.misc import NestedTensor, box_xyxy_to_cxcywh
from adet.modeling.clip.untils import tokenize
from adet.modeling.clip.models import CLIPTextContextEncoder, ContextDecoder, PromptGenerator
from detectron2.config import LazyCall as L


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        if 'pix_cls_score_map' in xs:
            pix_cls_score_map = xs.pop('pix_cls_score_map')
        else:
            pix_cls_score_map = None
        out: List[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos, pix_cls_score_map

class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


class CLIPMaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

        self.text_encoder = CLIPTextContextEncoder(
            # pretrained='weights/RN50.pt',
            pretrained=cfg.CLIP.WEIGHTS ,
            context_length=cfg.CLIP.TEXT_ENCODER.context_length, # len of clip text encoder input
            embed_dim=cfg.CLIP.TEXT_ENCODER.embed_dim,
            transformer_width=cfg.CLIP.TEXT_ENCODER.transformer_width,
            transformer_heads=cfg.CLIP.TEXT_ENCODER.transformer_heads,
            transformer_layers=cfg.CLIP.TEXT_ENCODER.transformer_layers,
            style=cfg.CLIP.TEXT_ENCODER.style)
        self.context_decoder = ContextDecoder(
            transformer_width=cfg.CLIP.CONTEXT_DECODER.transformer_width,
            transformer_heads=cfg.CLIP.CONTEXT_DECODER.transformer_heads,
            transformer_layers=cfg.CLIP.CONTEXT_DECODER.transformer_layers,
            visual_dim=cfg.CLIP.CONTEXT_DECODER.visual_dim,
            dropout=cfg.CLIP.CONTEXT_DECODER.dropout,
            outdim=cfg.CLIP.CONTEXT_DECODER.outdim,
            style=cfg.CLIP.CONTEXT_DECODER.style)

        self.prompt_generator = PromptGenerator(
            visual_dim=cfg.CLIP.PROMPT_GENERATOR.visual_dim,
            token_embed_dim=cfg.CLIP.PROMPT_GENERATOR.token_embed_dim,
            style=cfg.CLIP.PROMPT_GENERATOR.style)

        self.visual_prompt_generator = ContextDecoder(
            transformer_width=cfg.CLIP.VISUAL_PROMPT_GENERATOR.transformer_width,
            transformer_heads=cfg.CLIP.VISUAL_PROMPT_GENERATOR.transformer_heads,
            transformer_layers=cfg.CLIP.VISUAL_PROMPT_GENERATOR.transformer_layers,
            visual_dim=cfg.CLIP.VISUAL_PROMPT_GENERATOR.visual_dim,
            dropout=cfg.CLIP.VISUAL_PROMPT_GENERATOR.dropout,
            outdim=cfg.CLIP.VISUAL_PROMPT_GENERATOR.outdim,
            style=cfg.CLIP.VISUAL_PROMPT_GENERATOR.style)

        self.context_length = cfg.CLIP.context_length
        self.score_concat_index = cfg.CLIP.score_concat_index
        self.tau = cfg.CLIP.tau
        self.scale_matching_score_map= cfg.CLIP.scale_matching_score_map
        self.use_learnable_prompt = cfg.CLIP.use_learnable_prompt
        self.use_learnable_prompt_only = cfg.CLIP.use_learnable_prompt_only
        if self.use_learnable_prompt_only:
            assert self.use_learnable_prompt == self.use_learnable_prompt_only
        self.use_context_decoder = cfg.CLIP.use_context_decoder
        if not self.use_learnable_prompt: # original clip text encoder
            self.context_length = self.text_encoder.context_length

        class_names = cfg.CLIP.class_names
        # input of text encoder of clip
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(self.texts)

        token_embed_dim = cfg.CLIP.token_embed_dim
        text_dim = cfg.CLIP.text_dim
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

        self.use_score_map_concat_gamma = cfg.CLIP.use_score_map_concat_gamma
        self.score_map_concat_gamma = nn.Parameter(torch.ones([]) * 1e-3)  # for updating  pix-cla matching map and feat.

        self.is_v2 = cfg.CLIP.is_v2
        if self.is_v2:
            self.meta_query = nn.Parameter(torch.randn(1024))
            self.pix_w = nn.Parameter(torch.ones(1024, 512))
            
    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)

        text_embeddings, pix_cls_score_map = self.instance_class_matching(features)
        # TODO add to origin feature res4, or x
        if self.use_score_map_concat_gamma:
            features[self.score_concat_index] = features[self.score_concat_index] + self.score_map_concat_gamma * pix_cls_score_map
        else:
            features[self.score_concat_index] = features[self.score_concat_index] + pix_cls_score_map
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        features['pix_cls_score_map'] = pix_cls_score_map
        return features

    def instance_class_matching(self, x):
        ''' calculate instance-class prompt text matching score map '''

        visual_embeddings = x[self.score_concat_index]
        global_feat = torch.mean(visual_embeddings, dim=(2,3))

        B, C, H, W = visual_embeddings.shape
        # (B, C, 1+H*W)
        visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
                                   dim=2).permute(0, 2, 1)  # B, N, C

        # texts is set of classes name embedding, contexts is learnable prompt embedding
        prompt_gen = None
        if self.prompt_generator is not None: # text prompt generator
            if self.is_v2:
                prompt_gen = self.prompt_generator(self.meta_query.expand(B, -1)) # (B, C)
            else:
                prompt_gen = self.prompt_generator(global_feat) # (B, C)

        contexts = self.contexts if self.use_learnable_prompt else None # (1, N, C)
        # (B, K, C), 取text_embeddings最后时刻t的值作为输出(BKLC->BKC)
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
        
        # import pdb;pdb.set_trace()
        if self.visual_prompt_generator is not None:
            vis_prompt_diff = self.visual_prompt_generator(visual_embeddings.reshape(B, C, H * W).permute(0, 2, 1),
                                                           text_embeddings)
            vis_prompt_diff = vis_prompt_diff.permute(0, 2, 1).reshape(B, C, H, W)
            visual_embeddings = visual_embeddings + self.vis_gamma * vis_prompt_diff

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


    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                img_idx,
                : int(np.ceil(float(h) / self.feature_strides[idx])),
                : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    In addition to the post processing of detectron2, we add scalign for 
    bezier control points.
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    # results = d2_postprocesss(results, output_height, output_width, mask_threshold)

    # scale bezier points
    if results.has("beziers"):
        beziers = results.beziers
        # scale and clip in place
        h, w = results.image_size
        beziers[:, 0].clamp_(min=0, max=w)
        beziers[:, 1].clamp_(min=0, max=h)
        beziers[:, 6].clamp_(min=0, max=w)
        beziers[:, 7].clamp_(min=0, max=h)
        beziers[:, 8].clamp_(min=0, max=w)
        beziers[:, 9].clamp_(min=0, max=h)
        beziers[:, 14].clamp_(min=0, max=w)
        beziers[:, 15].clamp_(min=0, max=h)
        beziers[:, 0::2] *= scale_x
        beziers[:, 1::2] *= scale_y

    if results.has("polygons"):
        polygons = results.polygons
        polygons[:, 0::2] *= scale_x
        polygons[:, 1::2] *= scale_y

    return results


@META_ARCH_REGISTRY.register()
class TransformerDetector(nn.Module):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        d2_backbone = MaskedBackbone(cfg)
        N_steps = cfg.MODEL.TRANSFORMER.HIDDEN_DIM // 2
        self.test_score_threshold = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST
        self.use_polygon = cfg.MODEL.TRANSFORMER.USE_POLYGON
        backbone = Joiner(d2_backbone, PositionalEncoding2D(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels
        self.testr = TESTR(cfg, backbone)

        box_matcher, point_matcher = build_matcher(cfg)
        
        loss_cfg = cfg.MODEL.TRANSFORMER.LOSS
        weight_dict = {'loss_ce': loss_cfg.POINT_CLASS_WEIGHT, 'loss_ctrl_points': loss_cfg.POINT_COORD_WEIGHT, 'loss_texts': loss_cfg.POINT_TEXT_WEIGHT}
        enc_weight_dict = {'loss_bbox': loss_cfg.BOX_COORD_WEIGHT, 'loss_giou': loss_cfg.BOX_GIOU_WEIGHT, 'loss_ce': loss_cfg.BOX_CLASS_WEIGHT}
        if loss_cfg.AUX_LOSS:
            aux_weight_dict = {}
            # decoder aux loss
            for i in range(cfg.MODEL.TRANSFORMER.DEC_LAYERS - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()})
            # encoder aux loss
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in enc_weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        enc_losses = ['labels', 'boxes']
        dec_losses = ['labels', 'ctrl_points', 'texts']

        self.criterion = SetCriterion(self.testr.num_classes, box_matcher, point_matcher,
                                      weight_dict, enc_losses, dec_losses, self.testr.num_ctrl_points, 
                                      focal_alpha=loss_cfg.FOCAL_ALPHA, focal_gamma=loss_cfg.FOCAL_GAMMA)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        images = self.preprocess_image(batched_inputs)
        output = self.testr(images)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            ctrl_point_cls = output["pred_logits"]
            ctrl_point_coord = output["pred_ctrl_points"]
            text_pred = output["pred_texts"]
            results = self.inference(ctrl_point_cls, ctrl_point_coord, text_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            raw_ctrl_points = targets_per_image.polygons if self.use_polygon else targets_per_image.beziers
            gt_ctrl_points = raw_ctrl_points.reshape(-1, self.testr.num_ctrl_points, 2) / torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            gt_text = targets_per_image.text
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "ctrl_points": gt_ctrl_points, "texts": gt_text})
        return new_targets

    def inference(self, ctrl_point_cls, ctrl_point_coord, text_pred, image_sizes):
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []

        text_pred = torch.softmax(text_pred, dim=-1)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        for scores_per_image, labels_per_image, ctrl_point_per_image, text_per_image, image_size in zip(
            scores, labels, ctrl_point_coord, text_pred, image_sizes
        ):
            selector = scores_per_image >= self.test_score_threshold
            scores_per_image = scores_per_image[selector]
            labels_per_image = labels_per_image[selector]
            ctrl_point_per_image = ctrl_point_per_image[selector]
            text_per_image = text_per_image[selector]
            result = Instances(image_size)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            result.rec_scores = text_per_image
            ctrl_point_per_image[..., 0] *= image_size[1]
            ctrl_point_per_image[..., 1] *= image_size[0]
            if self.use_polygon:
                result.polygons = ctrl_point_per_image.flatten(1)
            else:
                result.beziers = ctrl_point_per_image.flatten(1)
            _, topi = text_per_image.topk(1)
            result.recs = topi.squeeze(-1)
            results.append(result)
        return results


@META_ARCH_REGISTRY.register()
class CLIPTransformerDetector(nn.Module):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        d2_backbone = CLIPMaskedBackbone(cfg)
        N_steps = cfg.MODEL.TRANSFORMER.HIDDEN_DIM // 2
        self.test_score_threshold = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST
        self.use_polygon = cfg.MODEL.TRANSFORMER.USE_POLYGON
        backbone = Joiner(d2_backbone, PositionalEncoding2D(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels
        self.testr = TESTR(cfg, backbone)

        self.pix_matching_resolution_div = cfg.CLIP.pix_matching_resolution_div
        self.tau = cfg.CLIP.tau

        box_matcher, point_matcher = build_matcher(cfg)

        loss_cfg = cfg.MODEL.TRANSFORMER.LOSS
        weight_dict = {'loss_ce': loss_cfg.POINT_CLASS_WEIGHT, 'loss_ctrl_points': loss_cfg.POINT_COORD_WEIGHT,
                       'loss_texts': loss_cfg.POINT_TEXT_WEIGHT, 'loss_pix_cls': cfg.CLIP.PIX_CLS_LOSS_WEIGHT}
        enc_weight_dict = {'loss_bbox': loss_cfg.BOX_COORD_WEIGHT, 'loss_giou': loss_cfg.BOX_GIOU_WEIGHT, 'loss_ce': loss_cfg.BOX_CLASS_WEIGHT}
        if loss_cfg.AUX_LOSS:
            aux_weight_dict = {}
            # decoder aux loss
            for i in range(cfg.MODEL.TRANSFORMER.DEC_LAYERS - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()})
            # encoder aux loss
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in enc_weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        enc_losses = ['labels', 'boxes']
        dec_losses = ['labels', 'ctrl_points', 'texts']

        self.criterion = SetCriterion(self.testr.num_classes, box_matcher, point_matcher,
                                      weight_dict, enc_losses, dec_losses, self.testr.num_ctrl_points,
                                      focal_alpha=loss_cfg.FOCAL_ALPHA, focal_gamma=loss_cfg.FOCAL_GAMMA)

        self.dice_loss = DiceLoss(eps=1e-6)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        images = self.preprocess_image(batched_inputs)
        output = self.testr(images)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets, _ = self.prepare_targets(gt_instances)
            # targets, gt_masks = self.prepare_targets1(gt_instances)
            loss_dict = self.criterion(output, targets)

            # clip pixel-class matching loss
            pix_cls_score_map = output['pix_cls_score_map'] / self.tau
            sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
            sem_seg = torch.stack(sem_seg,dim=0)
            # sem_seg = batched_inputs[0]['sem_seg'].to(self.device)
            rescale_size = sem_seg.shape[1:]
            sem_seg_mask = torch.ones(sem_seg.shape, device=self.device)
            pix_cls_score_map = F.interpolate(input=pix_cls_score_map, size=rescale_size, mode='bilinear', align_corners=False)
            # N, 1, H, W
            pix_cls_score_map = F.logsigmoid(pix_cls_score_map).exp()  # binary_class
            loss_prob = self.dice_loss(pix_cls_score_map, sem_seg, sem_seg_mask)
            loss_dict['loss_pix_cls'] = loss_prob

            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            ctrl_point_cls = output["pred_logits"]
            ctrl_point_coord = output["pred_ctrl_points"]
            text_pred = output["pred_texts"]
            results = self.inference(ctrl_point_cls, ctrl_point_coord, text_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        gt_masks = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            raw_ctrl_points = targets_per_image.polygons if self.use_polygon else targets_per_image.beziers
            gt_ctrl_points = raw_ctrl_points.reshape(-1, self.testr.num_ctrl_points, 2) / torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            gt_text = targets_per_image.text
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "ctrl_points": gt_ctrl_points, "texts": gt_text})
        return new_targets, gt_masks

    def prepare_targets1(self, targets):
        new_targets = []
        gt_masks = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            raw_ctrl_points = targets_per_image.polygons if self.use_polygon else targets_per_image.beziers
            gt_ctrl_points = raw_ctrl_points.reshape(-1, self.testr.num_ctrl_points, 2) / torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            gt_text = targets_per_image.text
            import pdb;pdb.set_trace()
            gt_mask_per_image = targets_per_image.gt_masks.crop_and_resize(
                targets_per_image.gt_boxes.tensor, (w/self.pix_matching_resolution_div, h/self.pix_matching_resolution_div)
            ).to(device=self.device)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "ctrl_points": gt_ctrl_points,
                                "texts": gt_text})
            gt_masks.append(gt_mask_per_image)
        gt_masks = cat(gt_masks, dim=0)
        return new_targets, gt_masks

    def prepare_gt_masks(self, gt_instances):
        gt_masks = []
        for instances_per_image in gt_instances:
            # if len(instances_per_image.gt_boxes.tensor) == 0:
            #     continue
            gt_mask_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.gt_boxes.tensor, self.pix_matching_resolution
            ).to(device=self.device)
            gt_masks.append(gt_mask_per_image)
        gt_masks = cat(gt_masks, dim=0)
        return gt_masks

        # gt_masks = gt_masks[gt_inds]
        # N = gt_masks.size(0)
        # gt_masks = gt_masks.view(N, -1)
        # mask_losses = F.binary_cross_entropy_with_logits(
        #     pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="none")
        # mask_loss = mask_losses.mean(dim=-1)


    def inference(self, ctrl_point_cls, ctrl_point_coord, text_pred, image_sizes):
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []

        text_pred = torch.softmax(text_pred, dim=-1)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        for scores_per_image, labels_per_image, ctrl_point_per_image, text_per_image, image_size in zip(
                scores, labels, ctrl_point_coord, text_pred, image_sizes
        ):
            selector = scores_per_image >= self.test_score_threshold
            scores_per_image = scores_per_image[selector]
            labels_per_image = labels_per_image[selector]
            ctrl_point_per_image = ctrl_point_per_image[selector]
            text_per_image = text_per_image[selector]
            result = Instances(image_size)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            result.rec_scores = text_per_image
            ctrl_point_per_image[..., 0] *= image_size[1]
            ctrl_point_per_image[..., 1] *= image_size[0]
            if self.use_polygon:
                result.polygons = ctrl_point_per_image.flatten(1)
            else:
                result.beziers = ctrl_point_per_image.flatten(1)
            _, topi = text_per_image.topk(1)
            result.recs = topi.squeeze(-1)
            results.append(result)
        return results
