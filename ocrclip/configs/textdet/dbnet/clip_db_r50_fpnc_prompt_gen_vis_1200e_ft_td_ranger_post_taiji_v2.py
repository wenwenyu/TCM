_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/det_models/clip_dbnet_r50att_prompt_gen_vis_fpnc_v2.py',
    '../../_base_/det_datasets/td_tr.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_736}}


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
    pretrained='/home/wwyu/code/OCRCP/ocrclip/pretrained/RN50.pt',
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

# load_from = '/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip/clip_db_r50_fpnc_prompt_gen_vis_20e_8x16_st_real3_pretrain_taiji_xu_0412_231639/epoch_11.pth'

# optimizer
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
evaluation = dict(interval=2, metric='hmean-iou',
                  save_best='0_td500_test_hmean-iou:hmean',
                  rule='greater')

log_config = dict(
    interval=10,
    hooks=[
        dict(type='DetailTextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])
