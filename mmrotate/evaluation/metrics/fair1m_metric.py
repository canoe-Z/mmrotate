import os
import os.path as osp
import re
import tempfile
import zipfile
from collections import OrderedDict, defaultdict
from typing import Optional, Sequence

import numpy as np
import torch
from mmcv.ops import nms_quadri, nms_rotated
from mmengine.logging import MMLogger

from mmrotate.evaluation import eval_rbbox_map
from mmrotate.registry import METRICS
from mmrotate.structures.bbox import rbox2qbox
from .dota_metric import DOTAMetric
import matplotlib.pyplot as plt


@METRICS.register_module()
class FAIR1MMetric(DOTAMetric):
    """FAIR1M evaluation metric."""
    default_prefix: Optional[str] = 'fair1m'

    def merge_results(self, results: Sequence[dict],
                      outfile_prefix: str) -> str:
        """Merge patches' predictions into full image's results and generate a
        zip file for DOTA online evaluation.

        You can submit it at:
        https://captain-whu.github.io/DOTA/evaluation.html

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the zip files. If the
                prefix is "somepath/xxx", the zip files will be named
                "somepath/xxx/xxx.zip".
        """
        collector = defaultdict(list)

        for idx, result in enumerate(results):
            img_id = result.get('img_id', idx)
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            ori_bboxes = bboxes.copy()
            if self.predict_box_type == 'rbox':
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
            elif self.predict_box_type == 'qbox':
                ori_bboxes[..., :] = ori_bboxes[..., :] + np.array(
                    [x, y, x, y, x, y, x, y], dtype=np.float32)
            else:
                raise NotImplementedError
            label_dets = np.concatenate(
                [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
                axis=1)
            collector[oriname].append(label_dets)

        id_list, dets_list = [], []
        for oriname, label_dets_list in collector.items():
            big_img_results = []
            label_dets = np.concatenate(label_dets_list, axis=0)
            labels, dets = label_dets[:, 0], label_dets[:, 1:]
            for i in range(len(self.dataset_meta['classes'])):
                if len(dets[labels == i]) == 0:
                    big_img_results.append(dets[labels == i])
                else:
                    try:
                        cls_dets = torch.from_numpy(dets[labels == i]).cuda()
                    except:  # noqa: E722
                        cls_dets = torch.from_numpy(dets[labels == i])
                    if self.predict_box_type == 'rbox':
                        nms_dets, _ = nms_rotated(cls_dets[:, :5],
                                                  cls_dets[:,
                                                           -1], self.iou_thr)
                    elif self.predict_box_type == 'qbox':
                        nms_dets, _ = nms_quadri(cls_dets[:, :8],
                                                 cls_dets[:, -1], self.iou_thr)
                    else:
                        raise NotImplementedError
                    big_img_results.append(nms_dets.cpu().numpy())
            id_list.append(oriname)
            dets_list.append(big_img_results)

        if osp.exists(outfile_prefix):
            raise ValueError(f'The outfile_prefix should be a non-exist path, '
                             f'but {outfile_prefix} is existing. '
                             f'Please delete it firstly.')
        os.makedirs(outfile_prefix)

        files = [
            osp.join(outfile_prefix, img_id + '.xml')
            for img_id in id_list
        ]
        file_objs = [open(f, 'w') for f in files]
        for f, dets_per_cls in zip(file_objs, dets_list):
            self._write_head(f)
            for cls, dets in zip(self.dataset_meta['classes'], dets_per_cls):
                if dets.size == 0:
                    continue
                th_dets = torch.from_numpy(dets)
                if self.predict_box_type == 'rbox':
                    rboxes, scores = torch.split(th_dets, (5, 1), dim=-1)
                    qboxes = rbox2qbox(rboxes)
                elif self.predict_box_type == 'qbox':
                    qboxes, scores = torch.split(th_dets, (8, 1), dim=-1)
                else:
                    raise NotImplementedError
                for qbox, score in zip(qboxes, scores):
                    self._write_obj(f, cls, qbox, str(score.item()))
            self._write_tail(f)

        for f in file_objs:
            f.close()

        zip_folder = osp.join(outfile_prefix, 'submission_zip')
        os.makedirs(zip_folder)
        zip_path = osp.join(zip_folder, 'test.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, os.path.join('test', osp.split(f)[-1]))

        return zip_path

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()
        if self.merge_patches:
            # convert predictions to txt format and dump to zip file
            zip_path = self.merge_results(preds, outfile_prefix)
            logger.info(f'The submission file save at {zip_path}')
            return eval_results
        else:
            # convert predictions to coco format and dump to json file
            _ = self.results2json(preds, outfile_prefix)
            if self.format_only:
                logger.info('results are saved in '
                            f'{osp.dirname(outfile_prefix)}')
                return eval_results

        if self.metric == 'mAP' or 'pr_curve':
            assert isinstance(self.iou_thrs, list)
            dataset_name = self.dataset_meta['classes']
            dets = [pred['pred_bbox_scores'] for pred in preds]

            mean_aps = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, all_results = eval_rbbox_map(
                    dets,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=self.use_07_metric,
                    box_type=self.predict_box_type,
                    dataset=dataset_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)

                if self.metric == 'pr_curve':
                    self.plot_pr_curve(class_results=all_results,
                                       iou_thr=iou_thr,
                                       selected_classes=('Boeing737', 'A350', 'Motorboat', 'Small Car', 'Van'))

            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        else:
            raise NotImplementedError
        return eval_results

    def _write_head(self,
                    f):
        head = """<?xml version="1.0" encoding="utf-8"?>
        <annotation>
            <source>
            <filename>placeholder_filename</filename>
            <origin>GF2/GF3</origin>
            </source>
            <research>
                <version>4.0</version>
                <provider>placeholder_affiliation</provider>
                <author>placeholder_authorname</author>
                <!--参赛课题 -->
                <pluginname>placeholder_direction</pluginname>
                <pluginclass>placeholder_suject</pluginclass>
                <time>2020-07-2020-11</time>
            </research>
            <size>
                <width>placeholder_width</width>
                <height>placeholder_height</height>
                <depth>placeholder_depth</depth>
            </size>
            <!--存放目标检测信息-->
            <objects>
        """
        f.write(head)

    def _write_obj(self,
                   f,
                   cls: str,
                   bbox,
                   conf: float):
        obj_str = """        <object>
                    <coordinate>pixel</coordinate>
                    <type>rectangle</type>
                    <description>None</description>
                    <possibleresult>
                        <name>palceholder_cls</name>                
                        <probability>palceholder_prob</probability>
                    </possibleresult>
                    <!--检测框坐标，首尾闭合的矩形，起始点无要求-->
                    <points>  
                        <point>palceholder_coord0</point>
                        <point>palceholder_coord1</point>
                        <point>palceholder_coord2</point>
                        <point>palceholder_coord3</point>
                        <point>palceholder_coord0</point>
                    </points>
                </object>
        """
        obj_xml = obj_str.replace("palceholder_cls", cls)
        obj_xml = obj_xml.replace("palceholder_prob", conf)
        obj_xml = obj_xml.replace(
            "palceholder_coord0", f'{bbox[0]:.2f}'+", "+f'{bbox[1]:.2f}')
        obj_xml = obj_xml.replace(
            "palceholder_coord1", f'{bbox[2]:.2f}'+", "+f'{bbox[3]:.2f}')
        obj_xml = obj_xml.replace(
            "palceholder_coord2", f'{bbox[4]:.2f}'+", "+f'{bbox[5]:.2f}')
        obj_xml = obj_xml.replace(
            "palceholder_coord3", f'{bbox[6]:.2f}'+", "+f'{bbox[7]:.2f}')
        f.write(obj_xml)

    def _write_tail(self,
                    f):
        tail = """    </objects>
        </annotation>
        """
        f.write(tail)

    def plot_pr_curve(self, class_results, iou_thr, selected_classes):
        selected_indices = [self.dataset_meta['classes'].index(
            cls) for cls in selected_classes]
        plt.figure(figsize=(8, 6))

        for idx, class_idx in enumerate(selected_indices):
            result = class_results[class_idx]
            recalls = result['recall']
            precisions = result['precision']
            num_dets = result['num_dets']

            for i in range(num_dets - 2, -1, -1):
                precisions[i] = np.maximum(precisions[i], precisions[i + 1])

            plt.plot(recalls, precisions,
                     label=self.dataset_meta['classes'][class_idx])
            plt.title(f'Precision Recall Curve Per Class @ IoU = {iou_thr}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')

        plt.legend()
        plt.savefig(f'{iou_thr}.pdf', format='pdf')
        #plt.savefig(f'{iou_thr}.png')
