_base_ = [
    # '../../_base_/schedules/schedule_adam_600e.py',
    '../../_base_/runtime_10e.py',
    '../../_base_/det_models/panet_r50att_prompt_gen_vis_fpem_ffm.py',
    '../../_base_/det_datasets/icdar2015_taiji.py',
    '../../_base_/det_pipelines/panet_pipeline.py'
]

model = {{_base_.model}}

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_icdar2015 = {{_base_.train_pipeline_icdar2015}}
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
    # pretrained='/home/wwyu/code/OCRCP/ocrclip/pretrained/RN50.pt',
    pretrained='/apdcephfs/private_v_fisherwyu/code/OCRCLIP/ocrclip/pretrained/RN50.pt',
    # pretrained='/apdcephfs/share_887471/interns/v_willwhua/wenwenyu/code/detcp/ocrclip//pretrained/RN50.pt',
    class_names=prompt_class_names,
    backbone=dict(
        input_resolution=736
    ),
    bbox_head=dict(
        type='PANHead',
        loss=dict(type='PANLoss', speedup_bbox_thr=-1),
        postprocessor=dict(type='PANPostprocessor', text_repr_type='quad')),
    identity_head=dict(bbce_loss=True)
)


load_from ='/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip/panet_clip_r50att_prompt_gen_vis_fpem_ffm_20e_st150k_real3_pretrain_taiji_hua_0412_110334/epoch_20.pth'

optimizer = dict(type='Ranger', lr=1e-3, weight_decay=0,
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                                 'text_encoder': dict(lr_mult=0.0),
                                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=True)


# runtime settings
total_epochs = 1200
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# checkpoint_config = dict(by_epoch=False, interval=5000)
checkpoint_config = dict(by_epoch=True, interval=300)
evaluation = dict(interval=20, metric='hmean-iou',
                  save_best='0_icdar2015_test_hmean-iou:hmean',
                  rule='greater')

log_config = dict(
    interval=10,
    hooks=[
        dict(type='DetailTextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])