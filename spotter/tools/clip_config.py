# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 4/22/22 4:39 PM

from detectron2.config import CfgNode as CN
from detectron2.config import LazyCall as L

def add_model_config(cfg: CN) -> None:
    _C = cfg

    _C.DATA_ROOT = "datasets"

    _C.CLIP = CN()
    _C.CLIP.WEIGHTS = "weights/RN50.pt"
    _C.CLIP.DEVICE = "cuda"
    _C.CLIP.PIX_CLS_LOSS_WEIGHT = 1.0
    _C.CLIP.pix_matching_resolution_div = 16 # res3 8, res4 16, res5 32

    _C.CLIP.context_length = 14
    _C.CLIP.score_concat_index = "res4"
    _C.CLIP.tau = 0.07
    _C.CLIP.scale_matching_score_map = False
    _C.CLIP.use_learnable_prompt = True
    _C.CLIP.use_learnable_prompt_only = False
    _C.CLIP.use_context_decoder = True
    _C.CLIP.class_names = ["the pixels of many arbitrary-shape text instances."]
    _C.CLIP.token_embed_dim = 512
    _C.CLIP.text_dim = 1024
    _C.CLIP.is_v2 = False

    TEXT_ENCODER = CN()
    TEXT_ENCODER.context_length=18
    TEXT_ENCODER.embed_dim=1024
    TEXT_ENCODER.transformer_width=512
    TEXT_ENCODER.transformer_heads=8
    TEXT_ENCODER.transformer_layers=12
    TEXT_ENCODER.style="pytorch"
    _C.CLIP.TEXT_ENCODER = TEXT_ENCODER

    CONTEXT_DECODER = CN()
    CONTEXT_DECODER.transformer_width = 256
    CONTEXT_DECODER.transformer_heads = 4
    CONTEXT_DECODER.transformer_layers = 3
    CONTEXT_DECODER.visual_dim = 1024
    CONTEXT_DECODER.dropout = 0.1
    CONTEXT_DECODER.outdim = 1024
    CONTEXT_DECODER.style = "pytorch"
    _C.CLIP.CONTEXT_DECODER = CONTEXT_DECODER

    PROMPT_GENERATOR =CN()
    PROMPT_GENERATOR.visual_dim = 1024
    PROMPT_GENERATOR.token_embed_dim = 512
    PROMPT_GENERATOR.style = "pytorch"
    _C.CLIP.PROMPT_GENERATOR = PROMPT_GENERATOR

    VISUAL_PROMPT_GENERATOR=CN()
    VISUAL_PROMPT_GENERATOR.transformer_width = 256
    VISUAL_PROMPT_GENERATOR.transformer_heads = 4
    VISUAL_PROMPT_GENERATOR.transformer_layers = 3
    VISUAL_PROMPT_GENERATOR.visual_dim = 1024
    VISUAL_PROMPT_GENERATOR.dropout = 0.1
    VISUAL_PROMPT_GENERATOR.outdim = 1024
    VISUAL_PROMPT_GENERATOR.style = "pytorch"
    _C.CLIP.VISUAL_PROMPT_GENERATOR = VISUAL_PROMPT_GENERATOR


def add_clip_config(cfg: CN) -> None:
    add_model_config(cfg)