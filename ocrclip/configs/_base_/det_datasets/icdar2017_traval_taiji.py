dataset_type = 'OCRCLIPDataset'
data_root = '/apdcephfs/share_887471/interns/v_fisherwyu/data/mmocr_det_data/icdar2017'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training.json',
    img_prefix=f'{data_root}/imgs',
    data_name='icdar2017_train',
    pipeline=None)

train2 = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_val.json',
    img_prefix=f'{data_root}/imgs',
    data_name='icdar2017_val',
    pipeline=None)

# mlt19_data_root = '/apdcephfs/share_887471/interns/v_fisherwyu/data/mmocr_det_data/mlt2019'
# test_mlt19 = dict(
#     type=dataset_type,
#     ann_file=f'{mlt19_data_root}/instances_training.json',
#     img_prefix=f'{mlt19_data_root}/imgs',
#     data_name='mlt2019_train',
#     pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test.json',
    img_prefix=f'{data_root}/imgs',
    data_name='icdar2017_test',
    pipeline=None)

train_list = [train, train2]

# test_list = [test_mlt19]
test_list = [test]
