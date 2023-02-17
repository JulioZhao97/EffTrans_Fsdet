# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseFewShotDataset
from .builder import build_dataloader, build_dataset
from .coco import COCO_SPLIT, FewShotCocoDataset
from .dataloader_wrappers import NWayKShotDataloader, NWayKShotSingleDataloader
from .dataset_wrappers import NWayKShotDataset, NWayKShotSingleDataset, QueryAwareDataset
from .pipelines import CropResizeInstance, GenerateMask
from .utils import NumpyEncoder, get_copy_dataset_type
from .voc import VOC_SPLIT, FewShotVOCDataset

from .voc_tb import VOC_SPLIT_TB, FewShotVOCDatasetTB

from .voc_tb_var import VOC_SPLIT_TB_VAR, FewShotVOCDatasetTBVAR

from .voc_food_var import VOC_SPLIT_FOOD_VAR, FewShotVOCDatasetFOODVAR

__all__ = [
    'build_dataloader', 'build_dataset', 'QueryAwareDataset',
    'NWayKShotDataset', 'NWayKShotDataloader', 'BaseFewShotDataset',
    'FewShotVOCDataset', 'FewShotCocoDataset', 'CropResizeInstance',
    'GenerateMask', 'NumpyEncoder', 'COCO_SPLIT', 'VOC_SPLIT',
    'get_copy_dataset_type', 'NWayKShotSingleDataloader'
]
