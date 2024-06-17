import os

from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .datasets.text import register_text_instances

# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

_PREDEFINED_SPLITS_TEXT = {
    # datasets with bezier annotations
    "totaltext_train": ("totaltext/train_images", "totaltext/train.json"),
    "totaltext_val": ("totaltext/test_images", "totaltext/test.json"),
    "ctw1500_word_train": ("CTW1500/ctwtrain_text_image", "CTW1500/annotations/train_ctw1500_maxlen100_v2.json"),
    "ctw1500_word_test": ("CTW1500/ctwtest_text_image","CTW1500/annotations/test_ctw1500_maxlen100.json"),
    "syntext1_train": ("syntext1/images", "syntext1/annotations/train.json"),
    "syntext2_train": ("syntext2/images", "syntext2/annotations/train.json"),
    "mltbezier_word_train": ("mlt2017/images","mlt2017/annotations/train.json"),
    "rects_train": ("ReCTS/ReCTS_train_images", "ReCTS/annotations/rects_train.json"),
    "rects_val": ("ReCTS/ReCTS_val_images", "ReCTS/annotations/rects_val.json"),
    "rects_test": ("ReCTS/ReCTS_test_images", "ReCTS/annotations/rects_test.json"),
    "art_train": ("ArT/rename_artimg_train", "ArT/annotations/abcnet_art_train.json"), 
    "lsvt_train": ("LSVT/rename_lsvtimg_train", "LSVT/annotations/abcnet_lsvt_train.json"), 
    "chnsyn_train": ("ChnSyn/syn_130k_images", "ChnSyn/annotations/chn_syntext.json"),
    "icdar2013_train": ("icdar2013/train_images", "icdar2013/ic13_train.json"),
    "icdar2015_train": ("icdar2015/train_images", "icdar2015/ic15_train.json"),
    "icdar2015_test": ("icdar2015/test_images", "icdar2015/ic15_test.json"),
    # datasets with polygon annotations
    "totaltext_poly_train": ("totaltext/train_images", "totaltext/train_poly.json"),
    "totaltext_poly_val": ("totaltext/test_images", "totaltext/test_poly.json"),
    "ctw1500_word_poly_train": ("CTW1500/ctwtrain_text_image", "CTW1500/annotations/train_poly.json"),
    "ctw1500_word_poly_test": ("CTW1500/ctwtest_text_image","CTW1500/annotations/test_poly.json"),
    "syntext1_poly_train": ("syntext1/images", "syntext1/annotations/train_poly.json"),
    "syntext2_poly_train": ("syntext2/images", "syntext2/annotations/train_poly.json"),
    "mltbezier_word_poly_train": ("mlt2017/images","mlt2017/annotations/train_poly.json"),
    "icdar2015_poly_train": ("icdar2015/train_images", "icdar2015/train_poly.json"),
    "icdar2015_poly_test": ("icdar2015/test_images", "icdar2015/test_poly.json"),
    "icdar2019_poly_train": ("icdar2019/train_images", "icdar2019/train_poly.json"),
    "textocr_poly_train": ("textocr/train_images", "textocr/annotations/train_poly.json"),
    "textocr_poly_val": ("textocr/train_images", "textocr/annotations/val_poly.json"),
}

metadata_text = {
    "thing_classes": ["text"], # for instance
    # "stuff_classes": ["bg"] + ["text"] # for semantic
    "stuff_classes":  ["text"] # for semantic
}


def register_all_coco(root="datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# True for open source;
# Internally at fb, we register them elsewhere
# if __name__.endswith(".builtin"):
#     # Assume pre-defined datasets live in `./datasets`.
#     _root = os.getenv("ADELAIDET_DATASETS", "datasets")
#     register_all_coco(_root)
# register_all_coco()
