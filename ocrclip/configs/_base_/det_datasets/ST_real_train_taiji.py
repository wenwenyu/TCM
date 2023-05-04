icdar_style_clip_dataset_type = 'OCRCLIPDataset'
lmdb_style_clip_dataset_type = 'OCRCLIPDetDataset'

classes = ['text']

# data_root = '/home/wwyu/data/mmocr_det_data'
data_root = '/apdcephfs/share_887471/interns/v_fisherwyu/data/mmocr_det_data'


train_ann_file1 = f'{data_root}/icdar2015/instances_training.json'
train_img_prefix1 = f'{data_root}/icdar2015/imgs'
train1 = dict(
    type=icdar_style_clip_dataset_type,
    ann_file=train_ann_file1,
    img_prefix=train_img_prefix1,
    data_name='icdar2015',
    pipeline=None)


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


train_ann_file3 = f'{data_root}/ctw1500/instances_training.json'
train_img_prefix3 = f'{data_root}/ctw1500/imgs'
train3 = {key: value for key, value in train1.items()}
train3['ann_file'] = train_ann_file3
train3['img_prefix'] = train_img_prefix3
train3['data_name'] = 'ctw1500'

train_ann_file4 = f'{data_root}/total_text/instances_training.json'
train_img_prefix4 = f'{data_root}/total_text/imgs'
train4 = {key: value for key, value in train1.items()}
train4['ann_file'] = train_ann_file4
train4['img_prefix'] = train_img_prefix4
train4['data_name'] = 'total_text'

test_ann_file1 = f'{data_root}/icdar2015/instances_test.json'
test_img_prefix1 = f'{data_root}/icdar2015/imgs'
test1 = {key: value for key, value in train1.items()}
test1['ann_file'] = test_ann_file1
test1['img_prefix'] = test_img_prefix1
test1['data_name'] = 'icdar2015_test'

test_ann_file2 = f'{data_root}/ctw1500/instances_test.json'
test_img_prefix2 = f'{data_root}/ctw1500/imgs'
test2 = {key: value for key, value in train1.items()}
test2['ann_file'] = test_ann_file2
test2['img_prefix'] = test_img_prefix2
test2['data_name'] = 'ctw1500_test'

test_ann_file3 = f'{data_root}/total_text/instances_test.json'
test_img_prefix3 = f'{data_root}/total_text/imgs'
test3 = {key: value for key, value in train1.items()}
test3['ann_file'] = test_ann_file3
test3['img_prefix'] = test_img_prefix3
test3['data_name'] = 'total_text_test'

test_ann_file4 = f'{data_root}/td_tr/instances_td500_test.json'
test_img_prefix4 = f'{data_root}/td_tr/td500'
test4 = {key: value for key, value in train1.items()}
test4['ann_file'] = test_ann_file4
test4['img_prefix'] = test_img_prefix4
test4['data_name'] = 'td_test'

train_list = [train1, train2, train3, train4]

test_list = [test1, test2, test3, test4]
