# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
from typing import List,  Union

import mmcv
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import get, get_local_path

from mmrotate.registry import DATASETS
from .dior import DIORDataset


@DATASETS.register_module()
class MAR20Dataset(DIORDataset):
    """DIOR dataset for detection.

    Args:
        ann_subdir (str): Subdir where annotations are.
            Defaults to 'Annotations/Oriented Bounding Boxes/'.
        file_client_args (dict): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        ann_type (str): Choose obb or hbb as ground truth.
            Defaults to `obb`.
    """

    METAINFO = {
        'classes':
        (
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
            'A8', 'A9', 'A10', 'A11', 'A12', 'A13',
            'A14', 'A15', 'A16', 'A17', 'A18', 'A19',
            'A20'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                    (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                    (175, 116, 175), (250, 0, 30), (165, 42, 42),
                    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0),
                    (120, 166, 157)]
    }

    def parse_data_info(self, img_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            img_info (dict): Raw image information, usually it includes
                `img_id`, `file_name`, and `xml_path`.

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = {}
        img_path = osp.join(self.data_prefix['img_path'],
                            img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['xml_path'] = img_info['xml_path']

        # deal with xml file
        with get_local_path(
                img_info['xml_path'],
                backend_args=self.backend_args) as local_path:
            raw_ann_info = ET.parse(local_path)
        root = raw_ann_info.getroot()

        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        else:
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, backend='cv2')
            width, height = img.shape[:2]
            del img, img_bytes

        data_info['height'] = height
        data_info['width'] = width

        instances = []
        for obj in root.findall('object'):
            instance = {}
            cls = obj.find('name').text
            label = self.cat2label[cls]
            if label is None:
                continue

            if self.ann_type == 'obb':
                bnd_box = obj.find('robndbox')
                polygon = np.array([
                    float(bnd_box.find('x_left_top').text),
                    float(bnd_box.find('y_left_top').text),
                    float(bnd_box.find('x_right_top').text),
                    float(bnd_box.find('y_right_top').text),
                    float(bnd_box.find('x_right_bottom').text),
                    float(bnd_box.find('y_right_bottom').text),
                    float(bnd_box.find('x_left_bottom').text),
                    float(bnd_box.find('y_left_bottom').text),
                ]).astype(np.float32)
            else:
                bnd_box = obj.find('bndbox')
                if bnd_box is None:
                    continue
                polygon = np.array([
                    float(bnd_box.find('xmin').text),
                    float(bnd_box.find('ymin').text),
                    float(bnd_box.find('xmax').text),
                    float(bnd_box.find('ymin').text),
                    float(bnd_box.find('xmax').text),
                    float(bnd_box.find('ymax').text),
                    float(bnd_box.find('xmin').text),
                    float(bnd_box.find('ymax').text)
                ]).astype(np.float32)

            ignore = False
            if self.bbox_min_size is not None:
                assert not self.test_mode
                if width < self.bbox_min_size or height < self.bbox_min_size:
                    ignore = True
            if ignore:
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = polygon
            instance['bbox_label'] = label
            instances.append(instance)

        data_info['instances'] = instances
        return data_info
