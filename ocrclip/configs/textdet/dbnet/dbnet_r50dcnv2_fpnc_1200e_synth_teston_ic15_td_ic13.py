_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/schedules/schedule_sgd_1200e.py',
    '../../_base_/det_models/dbnet_r50dcnv2_fpnc.py',
    # '../../_base_/det_datasets/icdar2015.py',
    # '../../_base_/det_datasets/night.py',
    '../../_base_/det_datasets/icdar2013_teston_ic15_td_taiji.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}
# test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1152}}

# load_from = 'pretrained/textdet/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth'
# load_from = '/home/wwyu/code/OCRCLIP/ocrclip/pretrained/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth'
load_from = '/apdcephfs/private_v_fisherwyu/code/OCRCLIP/ocrclip/pretrained/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth'

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

checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(interval=5, metric='hmean-iou')


log_config = dict(
    interval=5,
    hooks=[
        dict(type='DetailTextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])
