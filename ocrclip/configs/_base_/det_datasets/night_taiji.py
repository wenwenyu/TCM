icdar_style_clip_dataset_type = 'OCRCLIPDataset'
lmdb_style_clip_dataset_type = 'OCRCLIPDetDataset'


# data_root = '/home/wwyu/data/mmocr_det_data'
data_root = '/apdcephfs/share_887471/interns/v_fisherwyu/data/mmocr_det_data/Text_at_Night'

train_data_names = ['MLT17', 'MLT19', 'TextOCR', 'ICDAR2019-ArT', 'COCOTEXT', 'ICDAR2019-LSVT-0', 'ICDAR2019-LSVT-1']
test_data_names = ['MLT17', 'TextOCR', 'ICDAR2019-ArT', 'COCOTEXT', 'ICDAR2019-LSVT-0', 'ICDAR2019-LSVT-1']

# train_data_names = ['all']
# test_data_names = ['all']

train_list = []
test_list = []

for data_name in train_data_names:
    train_ann_file1 = f'{data_root}/train_gts/gt_jsons/instances_{data_name}.json'
    train_img_prefix1 = f'{data_root}/train_images'
    train1 = dict(
        type=icdar_style_clip_dataset_type,
        ann_file=train_ann_file1,
        img_prefix=train_img_prefix1,
        data_name=data_name,
        pipeline=None)
    train_list.append(train1)


for data_name in test_data_names:
    train_ann_file1 = f'{data_root}/test_gts/gt_jsons/instances_{data_name}_test.json'
    train_img_prefix1 = f'{data_root}/test_images'
    train1 = dict(
        type=icdar_style_clip_dataset_type,
        ann_file=train_ann_file1,
        img_prefix=train_img_prefix1,
        data_name=f'{data_name}_test',
        pipeline=None)
    test_list.append(train1)

test_list = train_list + test_list
