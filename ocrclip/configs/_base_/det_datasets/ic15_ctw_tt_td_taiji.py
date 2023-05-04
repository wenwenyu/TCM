dataset_type = 'OCRCLIPDataset'
data_root = '/apdcephfs/share_887471/interns/v_fisherwyu/data/mmocr_det_data'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/icdar2015/instances_training.json',
    img_prefix=f'{data_root}/icdar2015/imgs',
    data_name='icdar2015',
    pipeline=None)

test1 = dict(
    type=dataset_type,
    ann_file=f'{data_root}/icdar2015/instances_test.json',
    img_prefix=f'{data_root}/icdar2015/imgs',
    data_name='icdar2015_test',
    pipeline=None,
    postprocessor_cfg=dict(type='DBParamPostprocessor',
                           text_repr_type='quad',
                           mask_thr=0.3, # 0.3, 0.1
                           min_text_score=0.3, # 0.3, 0.5, 0.6, 0.62
                           min_text_width=5,   # 3, 5
                           unclip_ratio=1.5,   # 1.5, 1.7, 2
                           use_approxPolyDP=True,
                           arcLength_ratio=0.01, # 0.01
                           max_candidates=3000  # 1000, 3000
                           )
)


test2 = dict(
    type=dataset_type,
    ann_file=f'{data_root}/ctw1500/instances_test.json',
    img_prefix=f'{data_root}/ctw1500/imgs',
    data_name='ctw1500_test',
    pipeline=None,
    postprocessor_cfg=dict(type='DBParamPostprocessor',
                  text_repr_type='poly',
                  mask_thr=0.12, # 0.3
                  min_text_score=0.5, # 0.3
                  min_text_width=5,
                  unclip_ratio=1.9,
                  arcLength_ratio=0.001, # 0.01
                  max_candidates=3000  # 1000， 3000
                  )
)


test3 = dict(
    type=dataset_type,
    ann_file=f'{data_root}/total_text/instances_test.json',
    img_prefix=f'{data_root}/total_text/imgs',
    data_name='total_text_test',
    pipeline=None,
    postprocessor_cfg=dict(type='DBParamPostprocessor',
                  text_repr_type='poly',
                  mask_thr=0.33, # 0.3
                  min_text_score=0.53, # 0.3
                  min_text_width=5,
                  unclip_ratio=1.9,
                  arcLength_ratio=0.01, # 0.01
                  max_candidates=3000  # 1000， 3000
                  )
)


test4 = dict(
    type=dataset_type,
    ann_file=f'{data_root}/td_tr/instances_td500_test.json',
    img_prefix=f'{data_root}/td_tr/td500',
    data_name='td500_test',
    pipeline=None,
    postprocessor_cfg=dict(type='DBParamPostprocessor',
                  text_repr_type='quad',
                  mask_thr=0.08, # 0.3
                  min_text_score=0.17, # 0.3
                  min_text_width=5,
                  unclip_ratio=1.7,
                  arcLength_ratio=0.01, # 0.01
                  max_candidates=3000  # 1000， 3000
                  )
)


train_list = [train]

test_list = [test1, test2, test3, test4]
