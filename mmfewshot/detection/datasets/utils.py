# Copyright (c) OpenMMLab. All rights reserved.
import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Save numpy array obj to json."""

    def default(self, obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_copy_dataset_type(dataset_type: str) -> str:
    """Return corresponding copy dataset type."""
    if dataset_type in ['FewShotVOCDatasetTB', 'FewShotVOCDefaultDatasetTB']:
        copy_dataset_type = 'FewShotVOCCopyDatasetTB'
    elif dataset_type in ['FewShotVOCDatasetTBVAR', 'FewShotVOCDefaultDatasetTBVAR']:
        copy_dataset_type = 'FewShotVOCCopyDatasetTBVAR'
    elif dataset_type in ['FewShotVOCDatasetFOODVAR', 'FewShotVOCDefaultDatasetFOODVAR']:
        copy_dataset_type = 'FewShotVOCCopyDatasetFOODVAR'
    elif dataset_type in ['FewShotVOCDataset', 'FewShotVOCDefaultDataset']:
        copy_dataset_type = 'FewShotVOCCopyDataset'
    elif dataset_type in ['FewShotCocoDataset', 'FewShotCocoDefaultDataset']:
        copy_dataset_type = 'FewShotCocoCopyDataset'
    else:
        raise TypeError(f'{dataset_type} '
                        f'not support copy data_infos operation.')

    return copy_dataset_type
