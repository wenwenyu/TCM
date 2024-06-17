import logging
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import ProposalNetwork, GeneralizedRCNN
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.postprocessing import detector_postprocess as d2_postprocesss

from adet.modeling.testr.losses import SetCriterion, DiceLoss
from adet.modeling.clip.untils import tokenize
from adet.modeling.clip.models import CLIPTextContextEncoder, ContextDecoder, PromptGenerator

def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    In addition to the post processing of detectron2, we add scalign for 
    bezier control points.
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = d2_postprocesss(results, output_height, output_width, mask_threshold)

    # scale bezier points
    if results.has("beziers"):
        beziers = results.beziers
        # scale and clip in place
        beziers[:, 0::2] *= scale_x
        beziers[:, 1::2] *= scale_y
        h, w = results.image_size
        beziers[:, 0].clamp_(min=0, max=w)
        beziers[:, 1].clamp_(min=0, max=h)
        beziers[:, 6].clamp_(min=0, max=w)
        beziers[:, 7].clamp_(min=0, max=h)
        beziers[:, 8].clamp_(min=0, max=w)
        beziers[:, 9].clamp_(min=0, max=h)
        beziers[:, 14].clamp_(min=0, max=w)
        beziers[:, 15].clamp_(min=0, max=h)

    return results


@META_ARCH_REGISTRY.register()
class OneStageDetector(ProposalNetwork):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Uses "instances" as the return key instead of using "proposal".
    """
    def forward(self, batched_inputs):
        if self.training:
            return super().forward(batched_inputs)
        processed_results = super().forward(batched_inputs)
        processed_results = [{"instances": r["proposals"]} for r in processed_results]
        return processed_results


def build_top_module(cfg):
    top_type = cfg.MODEL.TOP_MODULE.NAME
    if top_type == "conv":
        inp = cfg.MODEL.FPN.OUT_CHANNELS
        oup = cfg.MODEL.TOP_MODULE.DIM
        top_module = nn.Conv2d(
            inp, oup,
            kernel_size=3, stride=1, padding=1)
    else:
        top_module = None
    return top_module



class CLIPPixClsBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""
    def __init__(self, cfg):
        super().__init__()
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
        self.score_concat_index = cfg.CLIP.score_concat_index # "res4"
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

        self.visual_embedding_proj = nn.Sequential(
            nn.Conv2d(256, text_dim, 1),
            nn.BatchNorm2d(text_dim),
            nn.ReLU(),
            nn.Conv2d(text_dim, text_dim, 1),
            nn.BatchNorm2d(text_dim)
        )
        
        self.is_v2 = cfg.CLIP.is_v2
        if self.is_v2:
            self.meta_query = nn.Parameter(torch.randn(1024))
            self.pix_w = nn.Parameter(torch.ones(1024, 512))
            
    def forward(self, features):
        # features = self.backbone(images.tensor)

        text_embeddings, pix_cls_score_map = self.instance_class_matching(features)
        # TODO add to origin feature res4, or x
        if self.use_score_map_concat_gamma:
            features[self.score_concat_index] = features[self.score_concat_index] + self.score_map_concat_gamma * pix_cls_score_map
        else:
            features[self.score_concat_index] = features[self.score_concat_index] + pix_cls_score_map
        features['pix_cls_score_map'] = pix_cls_score_map
        return features

    def instance_class_matching(self, x):
        ''' calculate instance-class prompt text matching score map '''

        visual_embeddings = x[self.score_concat_index]
        visual_embeddings = self.visual_embedding_proj(visual_embeddings)

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




@META_ARCH_REGISTRY.register()
class OneStageRCNN(GeneralizedRCNN):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.top_module = build_top_module(cfg)
        self.to(self.device)

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
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator( 
                images, features, gt_instances, self.top_module)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(
                    images, features, None, self.top_module)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return OneStageRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class CLIPOneStageRCNN(OneStageRCNN):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.clip_pix_cls = CLIPPixClsBackbone(cfg)
        self.loss_pix_cls_weight = cfg.CLIP.PIX_CLS_LOSS_WEIGHT
        self.pix_matching_resolution_div = cfg.CLIP.pix_matching_resolution_div
        self.tau = cfg.CLIP.tau
        self.dice_loss = DiceLoss(eps=1e-6)


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
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        # import pdb;pdb.set_trace()
        features = self.clip_pix_cls(features)
        pix_cls_score_map = features['pix_cls_score_map'] / self.tau

        bs = pix_cls_score_map.shape[0]
        loss_prob = []
        for i in range(bs):
            cur_pix_cls_score_map = pix_cls_score_map[i:i+1]
            sem_seg = batched_inputs[i]["sem_seg"].to(self.device)
            sem_seg = sem_seg.unsqueeze(0)
            # clip pixel-class matching loss
            # sem_seg = batched_inputs[0]['sem_seg'].to(self.device)
            rescale_size = sem_seg.shape[1:]
            sem_seg_mask = torch.ones(sem_seg.shape, device=self.device)
            cur_pix_cls_score_map = F.interpolate(input=cur_pix_cls_score_map, size=rescale_size, mode='bilinear', align_corners=False)
            # N, 1, H, W
            cur_pix_cls_score_map = F.logsigmoid(cur_pix_cls_score_map).exp()  # binary_class
            cur_loss_prob = self.dice_loss(cur_pix_cls_score_map, sem_seg, sem_seg_mask)
            loss_prob.append(cur_loss_prob)
        loss_prob = torch.stack(loss_prob, dim=0).mean()



        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances, self.top_module)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses['loss_pix_cls'] = loss_prob*self.loss_pix_cls_weight
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = self.clip_pix_cls(features)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(
                    images, features, None, self.top_module)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return OneStageRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
