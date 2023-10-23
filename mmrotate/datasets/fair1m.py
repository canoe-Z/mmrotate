import glob
import os.path as osp
from typing import List

from mmrotate.registry import DATASETS
from .dota import DOTADataset


@DATASETS.register_module()
class FAIR1MDataset(DOTADataset):
    METAINFO = {
        'classes':
        ('A220', 'A321', 'A330', 'A350', 'ARJ21', 'Baseball Field',
         'Basketball Court', 'Boeing737', 'Boeing747', 'Boeing777',
         'Boeing787', 'Bridge', 'Bus', 'C919', 'Cargo Truck',
         'Dry Cargo Ship', 'Dump Truck', 'Engineering Ship',
         'Excavator', 'Fishing Boat', 'Football Field', 'Intersection',
         'Liquid Cargo Ship', 'Motorboat', 'Passenger Ship', 'Roundabout',
         'Small Car', 'Tennis Court', 'Tractor', 'Trailer', 'Truck Tractor',
         'Tugboat', 'Van', 'Warship', 'other-airplane', 'other-ship',
         'other-vehicle'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(0, 0, 0), (66, 0, 75), (120, 0, 137), (130, 0, 147), (105, 0, 156), (30, 0, 166),
                    (0, 0, 187), (0, 0, 215), (0, 52, 221), (0, 119, 221),
                    (0, 137, 221), (0, 154, 215), (0, 164,
                                                   187), (0, 170, 162), (0, 170, 143),
                    (0, 164, 90), (0, 154, 15), (0, 168, 0),
                    (0, 186, 0), (0, 205, 0), (0, 224, 0), (0, 243, 0),
                    (41, 255, 0), (145, 255, 0), (203, 249, 0), (232, 239, 0),
                    (245, 222, 0), (255, 204, 0), (255,
                                                   175, 0), (255, 136, 0), (255, 51, 0),
                    (247, 0, 0), (228, 0, 0), (215, 0,
                                               0), (205, 0, 0), (204, 90, 90),
                    (204, 204, 204)]
    }

    def __init__(self,
                 **kwargs) -> None:
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in '
                                 f'{self.ann_file}')
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)

                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        instance = {}
                        bbox_info = si.split()
                        instance['bbox'] = [float(i) for i in bbox_info[:8]]
                        cls_name = ' '.join(bbox_info[8:-1])
                        instance['bbox_label'] = cls_map[cls_name]
                        difficulty = int(bbox_info[-1])
                        if difficulty > self.diff_thr:
                            instance['ignore_flag'] = 1
                        else:
                            instance['ignore_flag'] = 0
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list
