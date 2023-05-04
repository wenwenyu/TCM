dataset_type = 'OCRCLIPDataset'
data_root = '/apdcephfs/share_887471/interns/v_fisherwyu/data/mmocr_det_data/ctw1500'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training.json',
    img_prefix=f'{data_root}/imgs',
    data_name='ctw1500',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test.json',
    img_prefix=f'{data_root}/imgs',
    data_name='ctw1500_test',
    pipeline=None)

train_list = [train]

test_list = [test]
