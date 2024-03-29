
prompt_class_names = ['the pixels of many arbitrary-shape text instances.']
model = dict(
    type='OCRCLIP',
    pretrained='/apdcephfs/private_v_fisherwyu/code/OCRCLIP/ocrclip/pretrained/RN50.pt',
    context_length=14, # len of class name
    class_names=prompt_class_names,
    use_learnable_prompt=True,  # predefine text + learnable prompt
    use_learnable_prompt_only=False, # only use learnable prompt
    score_concat_index=3, # # start from 0 in range(backbone.out_indices)
    # backbone=dict(
    #     type='mmdet.ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='caffe'),
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        out_indices=(0, 1, 2, 3),
        output_dim=1024,
        input_resolution=640, # 512
        pix_text_match_stage_index=3,  # index of out_indices
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=18, # len of clip text encoder input
        embed_dim=1024,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    prompt_generator=dict(
        type='PromptGenerator',
        visual_dim=1024,
        token_embed_dim=512,
        style='pytorch'
    ),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=1024,
        dropout=0.1,
        outdim=1024,
        style='pytorch'),
    visual_prompt_generator=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=1024,
        dropout=0.1,
        outdim=1024,
        style='pytorch'),
    neck=dict(type='FPEM_FFM', in_channels=[256, 512, 1024, 2048+1]),
    bbox_head=dict(
        type='PANHead',
        in_channels=[128, 128, 128, 128],
        out_channels=6,
        loss=dict(type='PANLoss', speedup_bbox_thr=32),
        postprocessor=dict(type='PANPostprocessor', text_repr_type='poly')),
    identity_head=dict(
        type='PANIdentityHead',
        downsample_ratio=1/32.0,
        loss_weight=1.0,
        reduction='mean',
        negative_ratio=3.0,
        bbce_loss=False),
    train_cfg=None,
    test_cfg=None)
