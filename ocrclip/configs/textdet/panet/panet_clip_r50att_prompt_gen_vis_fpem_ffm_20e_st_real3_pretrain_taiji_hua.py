_base_ = [
    '../../_base_/runtime_10e.py',
    # '../../_base_/schedules/schedule_adam_600e.py',
    '../../_base_/det_models/panet_r50att_prompt_gen_vis_fpem_ffm.py',
    '../../_base_/det_datasets/ST_real_train_taiji_hua.py',
    '../../_base_/det_pipelines/panet_pipeline.py'
]

model = {{_base_.model}}

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

# train_pipeline_icdar2015 = {{_base_.train_pipeline_icdar2015}} # 736
train_pipeline_ctw1500 = {{_base_.train_pipeline_ctw1500}} # 640
# test_pipeline_icdar2015 = {{_base_.test_pipeline_icdar2015}}
test_pipeline_ctw1500 = {{_base_.test_pipeline_ctw1500}}

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_ctw1500),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_ctw1500),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_ctw1500))

prompt_class_names = ['the pixels of many arbitrary-shape text instances.']
model = dict(
    pretrained='/apdcephfs/share_887471/interns/v_willwhua/wenwenyu/code/detcp/ocrclip//pretrained/RN50.pt',
    # pretrained='/apdcephfs/private_v_fisherwyu/code/OCRCLIP/ocrclip/pretrained/RN50.pt',
    class_names=prompt_class_names,
    backbone=dict(
        input_resolution=640 # 512
    ),
    bbox_head=dict(
        type='PANHead',
        postprocessor=dict(type='PANPostprocessor', text_repr_type='poly')),
    identity_head=dict(bbce_loss=True)
)


# load_from =''

optimizer = dict(type='Ranger', lr=1e-3, weight_decay=0,
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                                 'text_encoder': dict(lr_mult=0.0),
                                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=True)


# runtime settings
total_epochs = 20
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# checkpoint_config = dict(by_epoch=False, interval=5000)
checkpoint_config = dict(by_epoch=True, interval=1)
evaluation = dict(interval=20, metric='hmean-iou')

log_config = dict(
    interval=10,
    hooks=[
        dict(type='DetailTextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])