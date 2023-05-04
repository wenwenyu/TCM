
# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)

prompt_class_names = ['the pixels of many arbitrary-shape text instances.']

model = dict(
    type='OCRCLIP',
    # pretrained='/home/wwyu/code/OCRCLIP/ocrclip/pretrained/ViT-B-16.pt',
    pretrained='/apdcephfs/private_v_fisherwyu/code/OCRCLIP/ocrclip/pretrained/ViT-B-16.pt',
    context_length=14,
    class_names=prompt_class_names,
    text_head=False,
    text_dim=512,
    score_concat_index=2,
    backbone=dict(
        type='CLIPVisionTransformer',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=640,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=18,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=512,
        dropout=0.1,
        outdim=512,
        style='pytorch'),
    neck=dict(
        type='FPNC',
        in_channels=[768, 768, 768+1, 768],
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
