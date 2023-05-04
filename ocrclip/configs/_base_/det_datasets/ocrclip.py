dataset_type = 'OCRCLIPDataset'
data_root = '/home/wwyu/data/mmocr_det_data/icdar2015'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training.json',
    img_prefix=f'{data_root}/imgs',
    data_name='icdar2015',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test.json',
    img_prefix=f'{data_root}/imgs',
    data_name='icdar2015_test',
    pipeline=None)

train_list = [train]

test_list = [test]
