dataset_type = 'VisOCRCLIPDataset'
data_root = '/home/wwyu/dataset/mmocr_det_data/td_tr'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_td500_train.json',
    img_prefix=f'{data_root}/td500',
    data_name='td500',
    pipeline=None)

train2 = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_tr400_train.json',
    img_prefix=f'{data_root}/tr400',
    data_name='tr400',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_td500_test.json',
    img_prefix=f'{data_root}/td500',
    data_name='td500_test',
    pipeline=None)

train_list = [train, train2]

test_list = [test]
