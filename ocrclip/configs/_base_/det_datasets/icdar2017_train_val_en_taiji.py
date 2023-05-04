dataset_type = 'OCRCLIPDataset'
data_root = '/apdcephfs/share_887471/interns/v_fisherwyu/data/mmocr_det_data/icdar2017'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training_en.json',
    img_prefix=f'{data_root}/imgs',
    data_name='icdar2017_en_train',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_val_en.json',
    img_prefix=f'{data_root}/imgs',
    data_name='icdar2017_en_val',
    pipeline=None)

train_test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training_val_en.json', # 694
    img_prefix=f'{data_root}/imgs',
    data_name='icdar2017_en_train_val',
    pipeline=None)

train_list = [train]

test_list = [train, test, train_test]
