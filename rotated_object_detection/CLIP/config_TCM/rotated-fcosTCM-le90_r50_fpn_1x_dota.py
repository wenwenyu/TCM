_base_ = [
    './TCM_dota.py', '../../configs/_base_/schedules/schedule_1x.py',
    '../../configs/_base_/default_runtime.py'
]
angle_version = 'le90'
IMAGE_MEAN = [v * 255 for v in (0.48145466, 0.4578275, 0.40821073)]
IMAGE_VAR = [v * 255 for v in (0.26862954, 0.26130258, 0.27577711)]
# model settings
ckpt_path = '/home/xkzhu/.cache/clip/RN50.pt'
num_cls = 15

model = dict(
    type='FCOS_TCM',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=IMAGE_MEAN,
        std=IMAGE_VAR,
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    seg_loss=True,
    context_length = 8,
    backbone=dict(
        type='VisualEncoder',
        pretrained_clip_path=ckpt_path,
        image_resolution=1024,
        freeze=False,
        ),
    text_encoder=dict(
        type='TextEncoder',
        pretrained_clip_path=ckpt_path,
        freeze=True
        ),
    prompt_generator=dict(
        type='PromptGenerator',
        visual_dim=1024,
        token_embed_dim=512,
        style='pytorch'
    ),
    vis_context_decoder=dict(
        type='ContextDecoder',
        pretrained_clip_path=ckpt_path,
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        dropout=0.1,
        ),
    text_context_decoder=dict(
        type='ContextDecoder',
        pretrained_clip_path=ckpt_path,
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        dropout=0.1,
        ),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048 + num_cls],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=num_cls,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        use_hbbox_loss=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_angle=None,
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

custom_imports = dict(
    imports=['CLIP'],
    allow_failed_imports=False)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001,
        eps=1e-8,  # fp16 to 1e-4,-3
        betas=(0.9, 0.999)),
        # Parameter-level learning rate and weight decay settings
        paramwise_cfg=dict(
            custom_keys={
                'backbone.visual': dict(lr_mult=0.3, decay_mult=1.0),  # lr_mult: 用于控制不同层的学习率，乘以基础学习率，
                'text_encoder': dict(lr_mult=0.0),
                'norm': dict(decay_mult=0.)
            }),
        clip_grad=dict(max_norm=0.1, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[9],
        gamma=0.1)
]

train_dataloader = dict(batch_size=4)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=2)