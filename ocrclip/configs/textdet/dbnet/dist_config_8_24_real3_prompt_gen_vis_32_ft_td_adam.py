_base_ = [
    '../../_base_/runtime_10e.py',
    # '../../_base_/schedules/schedule_sgd_1200e.py',
    # '../../_base_/det_models/dbnet_clipr50_fpnc.py',
    '../../_base_/det_models/clip_dbnet_r50att_prompt_gen_vis_fpnc.py',
    # '../../_base_/det_datasets/icdar2015.py',
    # '../../_base_/det_datasets/night.py',
    '../../_base_/det_datasets/td_tr_taiji.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
# test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}
test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_736}}
# test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1152}}


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


model = dict(
    pretrained='/apdcephfs/private_v_fisherwyu/code/OCRCLIP/ocrclip/pretrained/RN50.pt',
    # backbone=dict(
    #     pretrained='/home/wwyu/code/OCRCP/ocrclip/pretrained/RN50.pt'
    # ),
    # scale_matching_score_map=True,
    scale_matching_score_map=False,
    bbox_head=dict(
        type='DBFP16Head',
        loss=dict(bbce_loss=True),
        postprocessor=dict(type='DBVisPostprocessor',
                           text_repr_type='quad',
                           mask_thr=0.08, # 0.3
                           min_text_score=0.17, # 0.3
                           min_text_width=3,
                           return_vis_map=False,
                           use_approxPolyDP=False, ### False
                           unclip_ratio=1.9,
                           arcLength_ratio=0.01, # 0.01
                           max_candidates=1000  # 1000， 3000
                           )
    ),
    identity_head=dict(bbce_loss=True)
)

# load_from = 'pretrained/textdet/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth'
# load_from = '/home/wwyu/code/OCRCLIP/ocrclip/pretrained/textdet/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth'
# load_from = '/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip/clip_dbnet_r50_fpnc_prompt_20e_8x24_st_real3_pretrain_taiji_0115_003924/epoch_10.pth'
# load_from = '/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip/clip_db_r50_fpnc_prompt_gen_20e_8x16_st_real3_pretrain_taiji_0401_202141/latest.pth'
# load_from = '/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip/clip_db_r50_fpnc_prompt_gen_1200e_td_taiji_0408_034027/best_0_td500_test_hmean-iou:hmean_epoch_1100.pth'
# load_from = '/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip/clip_db_r50_fpnc_prompt_gen_vis_20e_8x16_st150k_real3_pretrain_taiji_xu_0411_224731/epoch_15.pth'
# load_from = '/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip/clip_db_r50_fpnc_prompt_gen_vis_20e_8x16_st150k_real3_pretrain_taiji_xu_0411_224731/epoch_20.pth'
# load_from = '/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip/clip_db_r50_fpnc_prompt_gen_vis_20e_8x16_st_real3_pretrain_taiji_xu_0412_231639/epoch_3.pth'
# load_from = '/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip/clip_db_r50_fpnc_prompt_gen_vis_20e_8x16_st_real3_pretrain_taiji_xu_0412_231639/epoch_7.pth'
# load_from = '/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip/clip_db_r50_fpnc_prompt_gen_vis_20e_8x16_st_real3_pretrain_taiji_resume_xu_0417_032903/epoch_13.pth'
# load_from = '/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip/clip_db_r50_fpnc_prompt_gen_vis_20e_8x16_st_real3_pretrain_taiji_xu_0412_231639/epoch_5.pth'
load_from = '/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip/clip_db_r50_fpnc_prompt_gen_vis_20e_8x16_st_real3_pretrain_taiji_xu_0412_231639/epoch_11.pth'

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
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=True)



# runtime settings
total_epochs = 1200
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# checkpoint_config = dict(by_epoch=False, interval=5000)
checkpoint_config = dict(by_epoch=True, interval=300)
evaluation = dict(interval=2, metric='hmean-iou',
                  save_best='0_td500_test_hmean-iou:hmean',
                  rule='greater')

log_config = dict(
    interval=10,
    hooks=[
        dict(type='DetailTextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])
