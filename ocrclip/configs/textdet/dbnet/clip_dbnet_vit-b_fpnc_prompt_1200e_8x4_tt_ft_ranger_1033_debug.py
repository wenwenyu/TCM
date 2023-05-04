_base_ = [
    '../../_base_/runtime_10e.py',
    # '../../_base_/schedules/schedule_adamw_2e.py',
    # '../../_base_/det_models/clip_dbnet_vit-b_fpnc.py',
    '../../_base_/det_models/clip_dbnet_vit-b_prompt_fpnc.py',
    # '../../_base_/det_datasets/ST_real_train_noval_taiji.py',
    '../../_base_/det_datasets/total_text_1033.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]


train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
# test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}
test_pipeline_4068_1024 = {{_base_.test_pipeline_800_800}}
# test_pipeline_4068_1024 = {{_base_.test_pipeline_vis_800_800}}

# load_from = 'pretrained/textdet/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth'
# load_from = '/home/wwyu/code/OCRCLIP/ocrclip/pretrained/textdet/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth'


data = dict(
    samples_per_gpu=2,
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
    pretrained='/home/wwyu/code/OCRCP/ocrclip/pretrained/ViT-B-16.pt',
    class_names=prompt_class_names,
    bbox_head=dict(
        type='DBFP16Head',
        loss=dict(bbce_loss=True),
        # postprocessor=dict(type='TextSegPostprocessor', # TextSegPostprocessor DBVisPostprocessor
        #                    text_repr_type='poly',
        #                    mask_thr=0.15, # 0.3
        #                    min_text_score=0.5, # 0.3
        #                    min_text_width=5,
        #                    with_logits=False,
        #                    return_vis_map=True,
        #                    use_approxPolyDP=False, # important
        #                    unclip_ratio=1.6,
        #                    arcLength_ratio=0.001, # 0.01
        #                    max_candidates=1000  # 1000， 3000
        #                    )

        postprocessor=dict(type='DBParamPostprocessor', # DBParamPostprocessor DBVisPostprocessor
                           text_repr_type='poly',
                           mask_thr=0.32, # 0.3, 0.1
                           min_text_score=0.35, # 0.3, 0.5, 0.62
                           min_text_width=3,   # 3, 5
                           with_logits=False,
                           return_vis_map=True,
                           use_approxPolyDP=False, # important
                           unclip_ratio=2.0,   # 1.5, 1.7, 2
                           arcLength_ratio=0.01, # 0.01
                           max_candidates=3000  # 1000, 3000
                           )
    ),
    identity_head=dict(bbce_loss=True)
)

#DBParamPostprocessor post_hyper mask_thr: 0.32 min_text_score: 0.35 min_text_width: 3 use_approxPolyDP:False arcLength_ratio: 0.01 unclip_ratio: 2.0 ckpt: epoch_200.pth max_candidates: 3000
# {'0_total_text_test_hmean-iou:recall': 0.781489841986456, '0_total_text_test_hmean-iou:precision': 0.8733602421796165, '0_total_text_test_hmean-iou:hmean': 0.8248749106504646}


#DBParamPostprocessor post_hyper mask_thr: 0.33 min_text_score: 0.53 min_text_width: 5 use_approxPolyDP:False arcLength_ratio: 0.01 unclip_ratio: 1.9 ckpt: epoch_200.pth max_candidates: 3000
# {'0_total_text_test_hmean-iou:recall': 0.7729119638826185, '0_total_text_test_hmean-iou:precision': 0.8963350785340314, '0_total_text_test_hmean-iou:hmean': 0.830060606060606}


#DBParamPostprocessor post_hyper mask_thr: 0.33 min_text_score: 0.53 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.01 unclip_ratio: 1.9 ckpt: epoch_200.pth max_candidates: 3000
# {'0_total_text_test_hmean-iou:recall': 0.7796839729119639, '0_total_text_test_hmean-iou:precision': 0.8806731259561448, '0_total_text_test_hmean-iou:hmean': 0.8271072796934866}


###DBParamPostprocessor post_hyper mask_thr: 0.35 min_text_score: 0.3 min_text_width: 5 use_approxPolyDP:False arcLength_ratio: 0.003 unclip_ratio: 1.94 ckpt: epoch_200.pth max_candidates: 3000
# {'0_total_text_test_hmean-iou:recall': 0.7747178329571106, '0_total_text_test_hmean-iou:precision': 0.8984293193717278, '0_total_text_test_hmean-iou:hmean': 0.8320000000000001}

#
# postprocessor=dict(type='DBVisPostprocessor',
#                    text_repr_type='poly',
#                    mask_thr=0.347, # 0.3, 0.1
#                    min_text_score=0.3, # 0.3, 0.5, 0.62
#                    min_text_width=5,   # 3, 5
#                    with_logits=False,
#                    return_vis_map=True,
#                    use_approxPolyDP=True, # important
#                    unclip_ratio=1.94,   # 1.5, 1.7, 2
#                    arcLength_ratio=0.003, # 0.01
#                    max_candidates=3000  # 1000, 3000
#                    )


# load_from='/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_vit-b_fpnc_prompt_20e_8x16_st_real3_pretrain_taiji_0227_232402/epoch_1.pth'
# load_from='/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_vit-b_fpnc_prompt_20e_8x16_st_real3_pretrain_taiji_0301_222906/latest.pth'
# load_from='/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_vit-b_fpnc_prompt_20e_8x16_st_real3_pretrain_taiji_0301_222906/epoch_10.pth'
# resume_from='/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_vit-b_fpnc_prompt_1200e_8x4_tt_ft_ranger_taiji_0306_205754/epoch_200.pth'

# fp16 = dict(loss_scale=512.)


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
checkpoint_config = dict(by_epoch=True, interval=50)
evaluation = dict(interval=50, metric='hmean-iou')


log_config = dict(
    interval=10,
    hooks=[
        dict(type='DetailTextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])