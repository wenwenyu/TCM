_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/schedules/schedule_sgd_1200e.py',
    '../../_base_/det_models/clip_dbnet_r50dcnv2_fpnc.py',
    '../../_base_/det_datasets/icdar2015_taiji.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]


train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=6,
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


model = dict(
    pretrained='/apdcephfs/share_887471/common/ocr_benchmark/fisherwwyu/OCRCLIP/ocrclip/pretrained/RN50.pt'
)

# load_from = '/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_r50_fpnc_20e_8x24_st_real3_pretrain_taiji/epoch_11.pth'
resume_from = '/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_r50_fpnc_1200e_8x24_ic15_ft_taiji/epoch_1170.pth'
# resume_from = '/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_r50_fpnc_1200e_8x24_ic15_ft_resume_taiji_0113_134038/epoch_1300.pth'
# optimizer

optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                                 'text_encoder': dict(lr_mult=0.0),
                                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=True)
# lr_config = dict(policy='ResumePoly', power=0.9, min_lr=1e-7, by_epoch=True)

# runtime settings
total_epochs = 1600
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# checkpoint_config = dict(by_epoch=False, interval=5000)
checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(interval=100, metric='hmean-iou')


log_config = dict(
    interval=5,
    hooks=[
        dict(type='DetailTextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])


