_base_ = [
    '../../_base_/runtime_10e.py',
    # '../../_base_/schedules/schedule_adamw_2e.py',
    # '../../_base_/det_models/clip_dbnet_r50dcnv2_fpnc.py',
    '../../_base_/det_models/clip_dbnet_r50att_prompt_fpnc.py',
    '../../_base_/det_datasets/total_text_1033.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]


train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
# test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}
# test_pipeline_4068_1024 = {{_base_.test_pipeline_800_800}}
test_pipeline_4068_1024 = {{_base_.test_pipeline_vis_800_800}}
# test_pipeline_4068_1024 = {{_base_.test_pipeline_2944_800}}

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


# prompt_class_names = ['text']
prompt_class_names = ['the pixels of many arbitrary-shape text instances.']
model = dict(
    pretrained='/apdcephfs/share_887471/common/ocr_benchmark/fisherwwyu/OCRCLIP/ocrclip/pretrained/RN50.pt',
    class_names=prompt_class_names,
    bbox_head=dict(
        type='DBFP16Head',
        loss=dict(bbce_loss=True),
        postprocessor=dict(type='DBVisPostprocessor',
                           text_repr_type='poly',
                           mask_thr=0.347, # 0.3, 0.1
                           min_text_score=0.3, # 0.3, 0.5, 0.62
                           min_text_width=5,   # 3, 5
                           with_logits=False,
                           return_vis_map=True,
                           use_approxPolyDP=True, # important
                           unclip_ratio=1.94,   # 1.5, 1.7, 2
                           arcLength_ratio=0.003, # 0.01
                           max_candidates=3000  # 1000, 3000
                           )
    ),
    identity_head=dict(bbce_loss=True)
)


###post_hyper mask_thr: 0.347 min_text_score: 0.3 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.005 unclip_ratio: 1.95 ckpt: epoch_250.pth max_candidates: 3000
# {'0_total_text_test_hmean-iou:recall': 0.8239277652370203, '0_total_text_test_hmean-iou:precision': 0.8732057416267942, '0_total_text_test_hmean-iou:hmean': 0.8478513356562137}

###############
###post_hyper mask_thr: 0.347 min_text_score: 0.3 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.004 unclip_ratio: 1.95 ckpt: epoch_250.pth max_candidates: 3000
###{'0_total_text_test_hmean-iou:recall': 0.8243792325056434, '0_total_text_test_hmean-iou:precision': 0.8736842105263158, '0_total_text_test_hmean-iou:hmean': 0.848315911730546}


###############
###post_hyper mask_thr: 0.347 min_text_score: 0.3 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.003 unclip_ratio: 1.95 ckpt: epoch_250.pth max_candidates: 3000
###{'0_total_text_test_hmean-iou:recall': 0.8243792325056434, '0_total_text_test_hmean-iou:precision': 0.8736842105263158, '0_total_text_test_hmean-iou:hmean': 0.848315911730546}


###post_hyper mask_thr: 0.347 min_text_score: 0.3 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.002 unclip_ratio: 1.95 ckpt: epoch_250.pth max_candidates: 3000
# {'0_total_text_test_hmean-iou:recall': 0.8225733634311513, '0_total_text_test_hmean-iou:precision': 0.8717703349282296, '0_total_text_test_hmean-iou:hmean': 0.8464576074332172}



##post_hyper mask_thr: 0.347 min_text_score: 0.3 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 2.1 ckpt: epoch_250.pth max_candidates: 3000
# {'0_total_text_test_hmean-iou:recall': 0.8207674943566592, '0_total_text_test_hmean-iou:precision': 0.8698564593301435, '0_total_text_test_hmean-iou:hmean': 0.8445993031358884}


##post_hyper mask_thr: 0.347 min_text_score: 0.3 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 2.0 ckpt: epoch_250.pth max_candidates: 3000
# {'0_total_text_test_hmean-iou:recall': 0.8230248306997743, '0_total_text_test_hmean-iou:precision': 0.8722488038277512, '0_total_text_test_hmean-iou:hmean': 0.8469221835075493}


##post_hyper mask_thr: 0.347 min_text_score: 0.3 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.97 ckpt: epoch_250.pth max_candidates: 3000
# {'0_total_text_test_hmean-iou:recall': 0.8225733634311513, '0_total_text_test_hmean-iou:precision': 0.8717703349282296, '0_total_text_test_hmean-iou:hmean': 0.8464576074332172}


########
##post_hyper mask_thr: 0.347 min_text_score: 0.3 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.95 ckpt: epoch_250.pth max_candidates: 3000
#{'0_total_text_test_hmean-iou:recall': 0.8234762979683973, '0_total_text_test_hmean-iou:precision': 0.8727272727272727, '0_total_text_test_hmean-iou:hmean': 0.8473867595818816}


##post_hyper mask_thr: 0.347 min_text_score: 0.3 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.93 ckpt: epoch_250.pth max_candidates: 3000
# {'0_total_text_test_hmean-iou:recall': 0.8234762979683973, '0_total_text_test_hmean-iou:precision': 0.8727272727272727, '0_total_text_test_hmean-iou:hmean': 0.8473867595818816}

##post_hyper mask_thr: 0.347 min_text_score: 0.3 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.92 ckpt: epoch_250.pth max_candidates: 3000
#  {'0_total_text_test_hmean-iou:recall': 0.8225733634311513, '0_total_text_test_hmean-iou:precision': 0.8717703349282296, '0_total_text_test_hmean-iou:hmean': 0.8464576074332172}


##post_hyper mask_thr: 0.35 min_text_score: 0.52 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.9 ckpt: epoch_250.pth max_candidates: 3000
#{'0_total_text_test_hmean-iou:recall': 0.8221218961625282, '0_total_text_test_hmean-iou:precision': 0.8704588910133844, '0_total_text_test_hmean-iou:hmean': 0.8456001857441374}

##post_hyper mask_thr: 0.348 min_text_score: 0.52 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.9 ckpt: epoch_250.pth max_candidates: 3000
#{'0_total_text_test_hmean-iou:recall': 0.8221218961625282, '0_total_text_test_hmean-iou:precision': 0.8712918660287081, '0_total_text_test_hmean-iou:hmean': 0.8459930313588849}

##post_hyper mask_thr: 0.347 min_text_score: 0.52 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.9 ckpt: epoch_250.pth max_candidates: 3000
#{'0_total_text_test_hmean-iou:recall': 0.8225733634311513, '0_total_text_test_hmean-iou:precision': 0.8717703349282296, '0_total_text_test_hmean-iou:hmean': 0.8464576074332172}

##post_hyper mask_thr: 0.346 min_text_score: 0.52 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.9 ckpt: epoch_250.pth max_candidates: 3000
#{'0_total_text_test_hmean-iou:recall': 0.8225733634311513, '0_total_text_test_hmean-iou:precision': 0.8717703349282296, '0_total_text_test_hmean-iou:hmean': 0.8464576074332172}

##post_hyper mask_thr: 0.345 min_text_score: 0.52 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.9 ckpt: epoch_250.pth max_candidates: 3000
# {'0_total_text_test_hmean-iou:recall': 0.8225733634311513, '0_total_text_test_hmean-iou:precision': 0.8717703349282296, '0_total_text_test_hmean-iou:hmean': 0.8464576074332172}

##post_hyper mask_thr: 0.343 min_text_score: 0.52 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.9 ckpt: epoch_250.pth max_candidates: 3000
#{'0_total_text_test_hmean-iou:recall': 0.8221218961625282, '0_total_text_test_hmean-iou:precision': 0.8717089516515079, '0_total_text_test_hmean-iou:hmean': 0.8461895910780669}


##post_hyper mask_thr: 0.34 min_text_score: 0.52 min_text_width: 5 use_approxPolyDP:True arcLength_ratio: 0.001 unclip_ratio: 1.9 ckpt: epoch_250.pth max_candidates: 3000
# {'0_total_text_test_hmean-iou:recall': 0.8216704288939052, '0_total_text_test_hmean-iou:precision': 0.8720651653090561, '0_total_text_test_hmean-iou:hmean': 0.8461180846118085}



##post_hyper mask_thr: 0.33 min_text_score: 0.52-0.6 min_text_width: 5 unclip_ratio: 1.9 ckpt: epoch_250.pth


# load_from = '/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_r50_fpnc_20e_8x24_st_real3_pretrain_taiji/epoch_11.pth'
# load_from = '/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_r50_fpnc_20e_8x24_st_real3_pretrain_taiji/epoch_7.pth'
# load_from = '/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_r50_fpnc_prompt_20e_8x24_st_real3_pretrain_taiji_0115_003924/epoch_18.pth'
load_from = '/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_r50_fpnc_prompt_20e_8x24_st_real3_pretrain_taiji_0115_003924/epoch_20.pth'


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
total_epochs = 1200
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
