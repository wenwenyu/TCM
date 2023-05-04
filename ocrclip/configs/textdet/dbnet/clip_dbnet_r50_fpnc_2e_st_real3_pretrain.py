_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/schedules/schedule_adamw_2e.py',
    '../../_base_/det_models/clip_dbnet_r50dcnv2_fpnc.py',
    # '../../_base_/det_datasets/ocrclip.py',
    # '../../_base_/det_datasets/ST_real_train.py',
    '../../_base_/det_datasets/test_det.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}

# load_from = 'pretrained/textdet/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth'
# load_from = '/home/wwyu/code/OCRCLIP/ocrclip/pretrained/textdet/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth'

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_r50dcnv2),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024))




# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                                 'text_encoder': dict(lr_mult=0.0),
                                                 'norm': dict(decay_mult=0.)}))
# optimizer = dict(type='SGD', lr=0.007, weight_decay=0.0001,
#                  paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
#                                                  'text_encoder': dict(lr_mult=0.0),
#                                                  'norm': dict(decay_mult=0.)}))

optimizer_config = dict(grad_clip=None)

# learning policy
# lr_config = dict(policy='poly', power=0.9)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=2)
checkpoint_config = dict(by_epoch=False, interval=5000)
# evaluation = dict(interval=100, metric='hmean-iou')
evaluation = dict(interval=1, metric='hmean-iou')
# total_epochs = 1200
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])