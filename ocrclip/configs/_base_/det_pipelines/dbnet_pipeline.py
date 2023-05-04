img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# `color`, `grayscale`, `unchanged`,
#             `color_ignore_orientation` and `grayscale_ignore_orientation`
train_pipeline_r18 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.5],
              dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.4),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]

train_pipeline_r18_shrink = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.5],
              dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.9),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]


test_pipeline_1333_736 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 736),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# for dbnet_r50dcnv2_fpnc
img_norm_cfg_r50dcnv2 = dict(
    mean=[122.67891434, 116.66876762, 104.00698793],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# CTW1500, ICDAR 2015/2017, and Totaltext dataset need color_type=color_ignore_orientation
train_pipeline_r50dcnv2 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg_r50dcnv2),
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.5],
              dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.4),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]

train_pipeline_r50dcnv2_shrink = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg_r50dcnv2),
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.5],
              dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.9),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]


test_pipeline_4068_1024 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# load annotation
test_pipeline_vis_4068_1024 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info']),
        ])
]

test_pipeline_4068_1152 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 1152),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline_vis_4068_1152 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 1152),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info']),
        ])
]


# td500
test_pipeline_736_736 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2944, 736),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# td500
test_pipeline_vis_736_736 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2944, 736),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info']),
        ])
]

# td500
test_pipeline_4068_736 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 736),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# td500
test_pipeline_vis_4068_736 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 736),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info']),
        ])
]

# tt
test_pipeline_800_800 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# tt
test_pipeline_vis_800_800 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info']),
        ])
]

# tt
test_pipeline_2944_800 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2944, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 800), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# tt
test_pipeline_4068_800 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 800), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


# tt
test_pipeline_vis_2944_800 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2944, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 800), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info']),
        ])
]

# ctw
test_pipeline_1600_1024 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1600, 1024), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


# ctw
test_pipeline_vis_1600_1024 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1600, 1024), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info']),
        ])
]

# icdar2017
test_pipeline_1024_768 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 768),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline_2944_768 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2944, 768),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# ic15, ctw, tt, td
test_pipeline_list = [test_pipeline_4068_1024, test_pipeline_4068_1024,
                      test_pipeline_800_800, test_pipeline_4068_736]