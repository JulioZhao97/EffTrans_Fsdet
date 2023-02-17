# Copyright (c) OpenMMLab. All rights reserved.
from .contrastive_bbox_head import ContrastiveBBoxHead
from .cosine_sim_bbox_head import CosineSimBBoxHead
from .meta_bbox_head import MetaBBoxHead
from .multi_relation_bbox_head import MultiRelationBBoxHead
from .two_branch_bbox_head import TwoBranchBBoxHead

# my addition
from .cosine_sim_bbox_head_my_tune import CosineSimBBoxHeadMyTune
from .cosine_sim_bbox_head_my_tune_test_ema import CosineSimBBoxHeadMyTuneTestEMA
from .cosine_sim_bbox_head_my_tune_test import CosineSimBBoxHeadMyTuneTest

from .cosine_sim_bbox_head_my_tune_coco import CosineSimBBoxHeadMyTuneCOCO

from .cosine_sim_bbox_head_my_tune_coco_test_ema import CosineSimBBoxHeadMyTuneCOCOTestEMA

from .cosine_sim_bbox_head_distill import CosineSimBBoxHeadDistill

# distill + tune
from .cosine_sim_bbox_head_my_tune_distill import CosineSimBBoxHeadMyTuneDistill

# TB
from .cosine_sim_bbox_head_my_tune_tb import CosineSimBBoxHeadMyTuneTB
from .cosine_sim_bbox_head_my_tune_test_ema_tb import CosineSimBBoxHeadMyTuneTestEMATB

__all__ = [
    'CosineSimBBoxHead', 'ContrastiveBBoxHead', 'MultiRelationBBoxHead',
    'MetaBBoxHead', 'TwoBranchBBoxHead'
]
