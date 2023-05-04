_base_ = [
    '../../_base_/runtime_10e.py',
    # '../../_base_/schedules/schedule_adamw_2e.py',
    # '../../_base_/det_models/clip_dbnet_r50att_prompt_fpnc.py',
    '../../_base_/det_models/clip_segnet_r50att_prompt_fpnc.py',
    # '../../_base_/det_datasets/ocrclip.py',
    '../../_base_/det_datasets/total_text_taiji.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]

# data_root = '/apdcephfs/share_887471/common/ocr_benchmark/mmocr_det_data'


train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

# train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2_shrink}}

# test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}
test_pipeline_4068_1024 = {{_base_.test_pipeline_800_800}}

# load_from = 'pretrained/textdet/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth'
# load_from = '/home/wwyu/code/OCRCLIP/ocrclip/pretrained/textdet/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth'
load_from='/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_segnet_r50_fpnc_prompt_20e_8x24_st_real3_pretrain_taiji_0306_025751/latest.pth'


data = dict(
    samples_per_gpu=6,
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



prompt_class_names = ['the pixels of many arbitrary-shape text instances.']
model = dict(
    pretrained='/apdcephfs/share_887471/common/ocr_benchmark/fisherwwyu/OCRCLIP/ocrclip/pretrained/RN50.pt',
    class_names=prompt_class_names,
    bbox_head=dict(
        type='TextSegHead',
        loss=dict(type='TextSegLoss', loss_weight=5.0, bbce_loss=True, with_logits=True,),
        postprocessor=dict(type='TextSegPostprocessor',
                           text_repr_type='poly',
                           mask_thr=0.3, # 0.3
                           min_text_score=0.5, # 0.3
                           min_text_width=5,
                           unclip_ratio=1.9,
                           with_logits=True,
                           arcLength_ratio=0.01, # 0.01
                           max_candidates=3000  # 1000， 3000
                           )

    ),
    identity_head=dict(loss_weight=1.0, bbce_loss=True)
)

# load_from='/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_segnet_r50_fpnc_prompt_20e_8x24_st_real3_pretrain_taiji_0306_025751/latest.pth'
load_from='/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_segnet_r50_fpnc_prompt_20e_8x24_st_real3_pretrain_taiji_0311_024052/latest.pth'



# optimizer
optimizer = dict(type='Ranger', lr=1e-3, weight_decay=0,
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                                 'text_encoder': dict(lr_mult=0.0),
                                                 'norm': dict(decay_mult=0.)}))

# optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001,
#                  paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
#                                                  'text_encoder': dict(lr_mult=0.0),
#                                                  'norm': dict(decay_mult=0.)}))
# optimizer = dict(type='SGD', lr=0.007, weight_decay=0.0001,
#                  paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
#                                                  'text_encoder': dict(lr_mult=0.0),
#                                                  'norm': dict(decay_mult=0.)}))

optimizer_config = dict(grad_clip=None)

# learning policy
# lr_config = dict(policy='poly', power=0.9)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=True)

# lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
#                  warmup='linear',
#                  warmup_iters=1500,
#                  warmup_ratio=1e-6)


# runtime settings
total_epochs = 1200
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(by_epoch=True, interval=300)
evaluation = dict(interval=50, metric='hmean-iou',
                  save_best='0_total_text_test_hmean-iou:hmean',
                  rule='greater')


log_config = dict(
    interval=10,
    hooks=[
        dict(type='DetailTextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])