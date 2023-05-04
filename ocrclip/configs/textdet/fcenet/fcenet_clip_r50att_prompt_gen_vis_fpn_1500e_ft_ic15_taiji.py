_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/det_models/fcenet_clip_r50att_prompt_gen_vis_fpn.py',
    '../../_base_/det_datasets/ST150K_real_train.py',
    '../../_base_/det_pipelines/fcenet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_icdar2015 = {{_base_.train_pipeline_clip_icdar2015}}
test_pipeline_icdar2015 = {{_base_.test_pipeline_icdar2015}}

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_icdar2015),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_icdar2015),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_icdar2015))

prompt_class_names = ['the pixels of many arbitrary-shape text instances.']
model = dict(
    pretrained='/home/wwyu/code/OCRCP/ocrclip/pretrained/RN50.pt',
    # pretrained='/home/wwyu/code/OCRCP/ocrclip/pretrained/RN50.pt',
    class_names=prompt_class_names,
    bbox_head=dict(
        type='FCEHead',
        postprocessor=dict(type='FCEPostprocessor',
                            text_repr_type='quad',
                            num_reconstr_points=50,
                            alpha=1.2,
                            beta=1.0,
                            score_thr=0.3
                           )
    ),
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
total_epochs = 1800
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# checkpoint_config = dict(by_epoch=False, interval=5000)
checkpoint_config = dict(by_epoch=True, interval=300)
evaluation = dict(interval=30, metric='hmean-iou',
                  save_best='0_icdar2015_test_hmean-iou:hmean',
                  rule='greater')


log_config = dict(
    interval=10,
    hooks=[
        dict(type='DetailTextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])