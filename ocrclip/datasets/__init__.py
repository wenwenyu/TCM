# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 12/25/21 8:37 PM

from .ocrclip_dataset import OCRCLIPDataset, OCRCLIPDetDataset, PostcfgUniformConcatDataset
from .pipelines import LoadImageWithPILFromFile
from .textdet_targets import FCENetCLIPTargets