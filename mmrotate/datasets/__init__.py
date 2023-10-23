# Copyright (c) OpenMMLab. All rights reserved.
from .dior import DIORDataset  # noqa: F401, F403
from .dota import DOTAv2Dataset  # noqa: F401, F403
from .dota import DOTADataset, DOTAv15Dataset
from .hrsc import HRSCDataset  # noqa: F401, F403
from .fair1m import FAIR1MDataset  # noqa: F401, F403
from .mar20 import MAR20Dataset  # noqa: F401, F403
from .transforms import *  # noqa: F401, F403

__all__ = [
    'DOTADataset', 'DOTAv15Dataset', 'DOTAv2Dataset', 'HRSCDataset',
    'DIORDataset', 'FAIR1MDataset', 'MAR20Dataset'
]
