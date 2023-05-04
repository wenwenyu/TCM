_base_ = [
    '../../_base_/runtime_10e.py',
    # '../../_base_/schedules/schedule_adamw_2e.py',
    # '../../_base_/det_models/clip_dbnet_r50dcnv2_fpnc.py',
    '../../_base_/det_models/clip_dbnet_r50att_prompt_fpnc.py',
    '../../_base_/det_datasets/icdar2015_1033.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]


train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}
# test_pipeline_4068_1024 = {{_base_.test_pipeline_1600_1024}}

data = dict(
    samples_per_gpu=4,
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

# prompt_class_names = ['text']
prompt_class_names = ['the pixels of many arbitrary-shape text instances.']
model = dict(
    pretrained='/home/wwyu/code/OCRCP/ocrclip/pretrained/RN50.pt',
    class_names=prompt_class_names,
    bbox_head=dict(
        type='DBFP16Head',
        loss=dict(bbce_loss=True),
        postprocessor=dict(type='DBVisPostprocessor',
                           text_repr_type='quad',
                           mask_thr=0.28, # 0.3
                           min_text_score=0.2, # 0.3
                           min_text_width=3,
                           with_logits=False,
                           return_vis_map=True,
                           use_approxPolyDP=True, # important
                           unclip_ratio=1.5,
                           arcLength_ratio=0.001, # 0.01
                           max_candidates=3000  # 1000， 3000
                           )
    ),
    identity_head=dict(bbce_loss=True)
)

###post_hyper DBParamPostprocessor mask_thr: 0.28 min_text_score: 0.2 min_text_width: 2-3 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.5 ckpt: epoch_600.pth max_candidates: 3000
# {'0_icdar2015_test_hmean-iou:recall': 0.8406355320173327, '0_icdar2015_test_hmean-iou:precision': 0.9326923076923077, '0_icdar2015_test_hmean-iou:hmean': 0.8842744998733857}


###post_hyper DBParamPostprocessor mask_thr: 0.28 min_text_score: 0.2 min_text_width: 2 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.6 ckpt: epoch_600.pth max_candidates: 3000
# {'0_icdar2015_test_hmean-iou:recall': 0.8415984593163216, '0_icdar2015_test_hmean-iou:precision': 0.9312733084709643, '0_icdar2015_test_hmean-iou:hmean': 0.8841679312089022}


#post_hyper DBParamPostprocessor mask_thr: 0.25 min_text_score: 0.2 min_text_width: 2 use_approxPolyDP:True arcLength_ratio: 0.0005 unclip_ratio: 1.6 ckpt: epoch_600.pth max_candidates: 3000
# {'0_icdar2015_test_hmean-iou:recall': 0.8387096774193549, '0_icdar2015_test_hmean-iou:precision': 0.9335476956055734, '0_icdar2015_test_hmean-iou:hmean': 0.8835911742328176}


###post_hyper DBParamPostprocessor mask_thr: 0.25 min_text_score: 0.2 min_text_width: 2 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.6 ckpt: epoch_600.pth max_candidates: 3000
# {'0_icdar2015_test_hmean-iou:recall': 0.8391911410688493, '0_icdar2015_test_hmean-iou:precision': 0.9340836012861736, '0_icdar2015_test_hmean-iou:hmean': 0.8840984022318031}


#post_hyper DBParamPostprocessor mask_thr: 0.3 min_text_score: 0.3 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.01 unclip_ratio: 1.6 ckpt: epoch_600.pth max_candidates: 3000
# {'0_icdar2015_test_hmean-iou:recall': 0.8363023591718826, '0_icdar2015_test_hmean-iou:precision': 0.9273892151628403, '0_icdar2015_test_hmean-iou:hmean': 0.8794936708860759}


# load_from = '/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_r50_fpnc_20e_8x24_st_real3_pretrain_taiji/epoch_11.pth'
# load_from = '/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_r50_fpnc_20e_8x24_st_real3_pretrain_taiji/epoch_20.pth'
# load_from = '/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_r50_fpnc_prompt_20e_8x24_st_real3_pretrain_taiji_0115_003924/epoch_18.pth'
# load_from = '/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_r50_fpnc_prompt_20e_8x24_st_real3_pretrain_taiji_0115_003924/epoch_20.pth'

# optimizer
optimizer = dict(type='Ranger', lr=1e-3, weight_decay=0,
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                                 'text_encoder': dict(lr_mult=0.0),
                                                 'norm': dict(decay_mult=0.)}))

# optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001,
#                  paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
#                                                  'text_encoder': dict(lr_mult=0.0),
#                                                  'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=True)

# runtime settings
total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# checkpoint_config = dict(by_epoch=False, interval=5000)
checkpoint_config = dict(by_epoch=True, interval=50)
evaluation = dict(interval=50, metric='hmean-iou')


log_config = dict(
    interval=10,
    hooks=[
        dict(type='DetailTextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])
