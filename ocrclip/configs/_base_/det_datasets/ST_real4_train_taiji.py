icdar_style_clip_dataset_type = 'OCRCLIPDataset'
lmdb_style_clip_dataset_type = 'OCRCLIPDetDataset'

classes = ['text']

# data_root = '/home/wwyu/data/mmocr_det_data'
data_root = '/apdcephfs/share_887471/interns/v_fisherwyu/data/mmocr_det_data'

train_data_names = ['icdar2015', 'ctw1500', 'total_text', 'icdar2017']
test_data_names = ['icdar2015', 'ctw1500', 'total_text', 'icdar2017']

train_list = []
test_list = []

for data_name in train_data_names:
    train_ann_file1 = f'{data_root}/{data_name}/instances_training.json'
    train_img_prefix1 = f'{data_root}/{data_name}/imgs'
    train1 = dict(
        type=icdar_style_clip_dataset_type,
        ann_file=train_ann_file1,
        img_prefix=train_img_prefix1,
        data_name=data_name,
        pipeline=None)
    train_list.append(train1)

train_ann_file2 = f'{data_root}/synthtext/instances_training.lmdb'
train_img_prefix2 = f'{data_root}/synthtext/imgs'
train2 = dict(
    type=lmdb_style_clip_dataset_type,
    data_name='synthtext',
    ann_file=train_ann_file2,
    img_prefix=train_img_prefix2,
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width',
                  'annotations'])),
    pipeline=None)

train_list.append(train2)

# train_ann_file3 = [f'{data_root}/icdar2017/instances_training.json', f'{data_root}/icdar2017/instances_val.json']
# train_img_prefix3 = f'{data_root}/icdar2017/imgs'
# train3 = {key: value for key, value in train1.items()}
# train3['ann_file'] = train_ann_file3
# train3['img_prefix'] = train_img_prefix3
# train3['data_name'] = 'icdar2017'
# train_list.append(train3)


for test_data_name in test_data_names:
    if test_data_name == 'icdar2017':
        test_ann_file1 = f'{data_root}/{test_data_name}/instances_val.json'
    else:
        test_ann_file1 = f'{data_root}/{test_data_name}/instances_test.json'
    test_img_prefix1 = f'{data_root}/{test_data_name}/imgs'
    test1 = dict(
        type=icdar_style_clip_dataset_type,
        ann_file=test_ann_file1,
        img_prefix=test_img_prefix1,
        data_name=f'{test_data_name}_test',
        pipeline=None)
    test_list.append(test1)


