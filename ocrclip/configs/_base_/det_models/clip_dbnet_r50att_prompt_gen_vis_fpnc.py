
# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)


# prompt_class_names = ['a set of many arbitrary-shape text instances.']
# prompt_class_names = ['the set of many arbitrary-shape text instances.']
prompt_class_names = ['the pixels of many arbitrary-shape text instances.']
# prompt_class_names = ['a set of texts, including horizontal, multi-oriented and curved text.']
# prompt_class_names = ['first text', 'second text']

model = dict(
    type='OCRCLIP',
    # pretrained='/home/wwyu/code/OCRCLIP/ocrclip/pretrained/RN50.pt',
    pretrained='/apdcephfs/private_v_fisherwyu/code/OCRCLIP/ocrclip/pretrained/RN50.pt',
    context_length=14, # len of class name
    class_names=prompt_class_names,
    use_learnable_prompt=True,  # predefine text + learnable prompt
    use_learnable_prompt_only=False, # only use learnable prompt
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        output_dim=1024,
        input_resolution=640, # 512
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
        type='FPNC',
        in_channels=[256, 512, 1024, 2048+1],
        lateral_channels=256
        ),
    bbox_head=dict(
        type='DBHead',
        in_channels=256,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=True),
        postprocessor=dict(type='DBPostprocessor', text_repr_type='quad')),
    identity_head=dict(
        type='IdentityHead',
        downsample_ratio=32.0,
        loss_weight=1.0,
        reduction='mean',
        negative_ratio=3.0,
        bbce_loss=True),
    # model training and testing settings
    train_cfg=None,
    test_cfg=None
)