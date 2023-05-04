# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 1/8/22 1:45 PM

import os.path as osp

from mmocr.datasets.builder import LOADERS
from mmocr.datasets.utils import LmdbLoader
from mmocr.utils import list_from_file



@LOADERS.register_module()
class LmdbFilterLoader(LmdbLoader):
    """Load annotation file with lmdb storage backend. Add filter file function"""
    def _load(self, ann_file):
        lmdb_anno_obj = LmdbAnnFileBackend(ann_file)

        return lmdb_anno_obj


class LmdbAnnFileBackend:
    """Lmdb storage backend for annotation file.

    Args:
        lmdb_path (str): Lmdb file path.
    """

    def __init__(self, lmdb_path, coding='utf8'):
        self.lmdb_path = lmdb_path
        self.coding = coding
        env = self._get_env()
        with env.begin(write=False) as txn:
            self.total_number = int(
                txn.get('total_number'.encode(self.coding)).decode(
                    self.coding))

    def __getitem__(self, index):
        """Retrieval one line from lmdb file by index."""
        # only attach env to self when __getitem__ is called
        # because env object cannot be pickle
        if not hasattr(self, 'env'):
            self.env = self._get_env()

        with self.env.begin(write=False) as txn:
            line = txn.get(str(index).encode(self.coding)).decode(self.coding)
        return line

    def __len__(self):
        return self.total_number

    def _get_env(self):
        import lmdb
        return lmdb.open(
            self.lmdb_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
