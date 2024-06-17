_base_ = [
    './TCM_dota.py', '../../configs/_base_/schedules/schedule_1x.py',
    '../../configs/_base_/default_runtime.py'
]
angle_version = 'le90'
IMAGE_MEAN = [v * 255 for v in (0.48145466, 0.4578275, 0.40821073)]
IMAGE_VAR = [v * 255 for v in (0.26862954, 0.26130258, 0.27577711)]
# model settings
ckpt_path = '/home/xkzhu/.cache/clip/RN50.pt'  #RN50x64.pt
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
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RotatedATSSHead',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='FakeRotatedAnchorGenerator',
            angle_version=angle_version,
            octave_base_scale=4,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder',
            angle_version=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='RotatedATSSAssigner',
            topk=9,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        sampler=dict(
            type='mmdet.PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
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
find_unused_parameters=True
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=12)