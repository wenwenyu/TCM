
prompt_class_names = ['the pixels of many arbitrary-shape text instances.']
model = dict(
    type='FCECLIP',
    pretrained='/apdcephfs/private_v_fisherwyu/code/OCRCLIP/ocrclip/pretrained/RN50.pt',
    context_length=14, # len of class name
    class_names=prompt_class_names,
    use_learnable_prompt=True,  # predefine text + learnable prompt
    use_learnable_prompt_only=False, # only use learnable prompt
    score_concat_index=2, # # start from 0 in range(backbone.out_indices)
    # backbone=dict(
    #     type='mmdet.ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(1, 2, 3),
    #     frozen_stages=-1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    #     norm_eval=False,
    #     style='pytorch'),
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        out_indices=(1, 2, 3),
        output_dim=1024,
        input_resolution=800, # 512
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
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048+1],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True,
        act_cfg=None),
    bbox_head=dict(
        type='FCEHead',
        in_channels=256,
        scales=(8, 16, 32),
        fourier_degree=5,
        loss=dict(type='FCELoss', num_sample=50),
        postprocessor=dict(
            type='FCEPostprocessor',
            text_repr_type='quad',
            num_reconstr_points=50,
            alpha=1.2,
            beta=1.0,
            score_thr=0.3)),
    identity_head=dict(
        type='FCEIdentityHead',
        downsample_ratio=32.0,
        loss_weight=1.0,
        reduction='mean',
        negative_ratio=3.0,
        bbce_loss=True)
)
