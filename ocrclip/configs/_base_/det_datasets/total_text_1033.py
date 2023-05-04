dataset_type = 'OCRCLIPDataset'
data_root = '/home/wwyu/dataset/mmocr_det_data/total_text'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training.json',
    img_prefix=f'{data_root}/imgs',
    data_name='total_text',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test.json',
    img_prefix=f'{data_root}/imgs',
    data_name='total_text_test',
    pipeline=None)

train_list = [train]

test_list = [test]
