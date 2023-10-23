import copy
from typing import List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import batched_nms
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import (multi_apply, images_to_levels, unmap)
from mmdet.structures.bbox import (
    BaseBoxes, cat_boxes, get_box_tensor, scale_boxes)
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmdet.models.task_modules.prior_generators import anchor_inside_flags
from mmdet.models.task_modules.samplers import PseudoSampler
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from mmengine.model import bias_init_with_prob, normal_init
from torch import Tensor

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures.bbox import rbox2hbox
from mmrotate.structures.bbox.rotated_boxes import RotatedBoxes


class DecoupledAttentionModule(nn.Module):
    """Layer and spatial attention in QOPN.

    Args:
        feat_channels (int): Number of feature channels in qopn head.
        stacked_convs (int): Number of conv layers in qopn head.
        la_down_rate (int): Downsample rate of layer attention.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self,
                 feat_channels: int,
                 stacked_convs: int,
                 enable_sa: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None) -> None:
        super().__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.enable_sa = enable_sa
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Conv2d(
            self.in_channels,
            self.stacked_convs,
            1,
            stride=1,
            padding=0,
            groups=1,
            bias=True)

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=norm_cfg is None)

        if self.enable_sa:
            self.spatial_attention = nn.Conv2d(
                2,
                1,
                7,
                stride=1,
                padding=3,
                groups=1,
                bias=True)
            self.gamma = nn.Parameter(
                1e-6 * torch.ones((1, feat_channels, 1, 1)), requires_grad=True)

    def init_weights(self) -> None:
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        if self.enable_sa:
            for m in self.spatial_attention.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self,
                feat: Tensor,
                avg_feat: Optional[Tensor] = None) -> Tensor:
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs.
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)

        if self.enable_sa:
            avg_feat = torch.mean(feat, dim=1, keepdim=True)
            max_feat, _ = torch.max(feat, dim=1, keepdim=True)
            spatial_weight = torch.cat([avg_feat, max_feat], dim=1)
            spatial_weight = self.spatial_attention(spatial_weight)
            feat = feat + self.gamma * feat * spatial_weight

        return feat


@MODELS.register_module()
class QualityOrientedRPNHead(AnchorFreeHead):
    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 start_level: int = 0,
                 angle_version: str = 'le90',
                 anchor_generator: ConfigType = dict(
                     type='FakeRotatedAnchorGenerator',
                     octave_base_scale=8,
                     scales_per_octave=1,
                     ratios=[1.0],
                     strides=[8, 16, 32, 64, 128]),
                 scale_angle: bool = False,
                 enable_dam: bool = True,
                 use_fpn_feature: bool = True,
                 enable_sa: bool = True,
                 bbox_coder: ConfigType = dict(type='DistanceAnglePointCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.5),
                 loss_bbox: ConfigType = dict(
                     type='RotatedIoULoss', loss_weight=0.5),
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.start_level = start_level
        self.scale_angle = scale_angle
        self.angle_version = angle_version
        self.enable_dam = enable_dam
        self.use_fpn_feature = use_fpn_feature
        self.enable_sa = enable_sa
        assert num_classes == 1
        super().__init__(
            num_classes=1,
            in_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            bbox_coder=bbox_coder,
            norm_cfg=norm_cfg,
            **kwargs)

        self.prior_generator = TASK_UTILS.build(anchor_generator)

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = PseudoSampler(context=self)

        # Usually the numbers of anchors for each level are the same
        # except SSD detectors. So it is an int in the most dense
        # heads but a list of int in SSDHead
        self.num_base_priors = self.prior_generator.num_base_priors[0]

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.inter_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.inter_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        if self.use_fpn_feature:
            num_layers = self.stacked_convs + 1
        else:
            num_layers = self.stacked_convs
        if self.enable_dam:
            self.cls_decomp = DecoupledAttentionModule(self.feat_channels,
                                                       num_layers,
                                                       self.enable_sa,
                                                       self.conv_cfg, self.norm_cfg)
            self.reg_decomp = DecoupledAttentionModule(self.feat_channels,
                                                       num_layers,
                                                       self.enable_sa,
                                                       self.conv_cfg, self.norm_cfg)
        self.relu = nn.ReLU()
        self.qopn_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.qopn_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)
        self.qopn_angle = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.scale_angle:
            self.scale_angle = Scale(1.0)

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.075)
        for m in self.inter_convs:
            normal_init(m.conv, std=0.01)

        if self.enable_dam:
            self.cls_decomp.init_weights()
            self.reg_decomp.init_weights()

        normal_init(self.qopn_cls, std=0.01, bias=bias_cls)
        normal_init(self.qopn_reg, std=0.01)
        normal_init(self.qopn_angle, std=0.01)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats[self.start_level:], self.scales, self.strides)

    def forward_single(self, x: Tensor, scale: Scale, stride: int) -> Sequence[Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        inter_feats = []
        if self.use_fpn_feature:
            inter_feats.append(self.relu(x))
        for inter_conv in self.inter_convs:
            x = inter_conv(x)
            inter_feats.append(x)
        feat = torch.cat(inter_feats, 1)

        if self.enable_dam:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)
            reg_feat = self.reg_decomp(feat, avg_feat)
        else:
            cls_feat = inter_feats[-1]
            reg_feat = inter_feats[-1]

        cls_score = self.qopn_cls(cls_feat)
        bbox_pred = scale(self.qopn_reg(reg_feat).exp()).float() * stride
        angle_pred = self.qopn_angle(reg_feat)
        if self.scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()
        bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)

        return cls_score, bbox_pred

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            anchors: Tensor, labels: Tensor,
                            bbox_targets: Tensor, num_total_samples: int):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 5).
            alignment_metrics (Tensor): Alignment metrics with shape
                (N, num_total_anchors).
            stride (tuple[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Equivalent substitution of ``@force_fp32()``
        cls_score = cls_score.float()
        bbox_pred = bbox_pred.float()

        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        anchors = anchors.reshape(-1, anchors.size(-1))
        target_dim = bbox_targets.size(-1)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, target_dim)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        labels = labels.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_prior_points = self.anchor_center(anchors[pos_inds])
            pos_decode_bbox_pred = self.bbox_coder.decode(pos_prior_points,
                                                          pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_prior_points, pos_bbox_targets)

            # regression loss
            centerness_targets = self.centerness_target(pos_bbox_targets)

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        loss_cls = self.loss_cls(
            cls_score, labels, avg_factor=num_total_samples)

        return loss_cls, loss_bbox, centerness_targets.sum()

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(cls_scores) == len(bbox_preds)

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, labels_weight_list,
         bbox_targets_list, num_total_pos) = cls_reg_targets

        num_total_samples = num_total_pos

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, bbox_avg_factor = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            bbox_targets_list,
            num_total_samples=num_total_samples)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(
            loss_rpn_cls=losses_cls,
            loss_rpn_bbox=losses_bbox)

    def _get_targets_single(self,
                            flat_anchors: Union[Tensor, BaseBoxes],
                            valid_flags: Tensor,
                            num_level_anchors: List[int],
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (torch.Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape \
                (num_anchors, 5)
            valid_flags (torch.Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of \
                shape (num_anchors,).
            num_level_anchors (torch.Tensor): Number of anchors of each \
                scale level
            gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
                shape (num_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be \
                ignored, shape (num_ignored_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_labels (torch.Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original \
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of all anchor
                label_weights_list (list[Tensor]): Label weights of all anchor
                bbox_targets_list (list[Tensor]): BBox targets of all anchor
                pos_inds (int): Indices of positive anchor
                neg_inds (int): Indices of negative anchor
                sampling_result: object `SamplingResult`, sampling result.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances,
                                             num_level_anchors_inside,
                                             gt_instances, gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors.tensor)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                self.anchor_center(sampling_result.pos_bboxes), sampling_result.pos_gt_bboxes.tensor)
            bbox_targets[pos_inds, :] = pos_bbox_targets

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        return (labels, label_weights, bbox_targets, pos_inds)

    def get_targets(self,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True) -> tuple:
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each \
                image. The outer list indicates images, and the inner list \
                corresponds to feature levels of the image. Each element of \
                the inner list is a tensor of shape (num_anchors, 5).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of \
                each image. The outer list indicates images, and the inner \
                list corresponds to feature levels of the image. Each element \
                of the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be \
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original \
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of \
                    each level
                - bbox_targets_list (list[Tensor]): BBox targets of each level
                - bbox_weights_list (list[Tensor]): BBox weights of each level
                - num_total_pos (int): Number of positive samples in all images
                - num_total_neg (int): Number of negative samples in all images
            additional_returns: This function enables user-defined returns \
                from self._get_targets_single`. These returns are currently \
                refined to properties at each feature map (HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        (all_labels, all_label_weights, all_bbox_targets,
         pos_inds_list) = multi_apply(
             self._get_targets_single,
             concat_anchor_list,
             concat_valid_flag_list,
             num_level_anchors_list,
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             unmap_outputs=unmap_outputs)
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        return (labels_list, label_weights_list,
                bbox_targets_list, num_total_pos)

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has
                shape (num_anchors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RPN head does not need this value.
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_anchors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        level_ids = []
        for level_idx, (cls_score, bbox_pred, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              mlvl_priors)):

            # Equivalent substitution of ``@force_fp32()``
            cls_score = cls_score.float()
            bbox_pred = bbox_pred.float()

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0] since mmdet v2.0
                # BG cat_id: 1
                scores = cls_score.softmax(-1)[:, :-1]

            scores = torch.squeeze(scores)
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                bbox_pred = bbox_pred[topk_inds, :]
                priors = priors[topk_inds]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(self.anchor_center(
            priors), bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.level_ids = torch.cat(level_ids)

        return self._bbox_post_process(
            results=results, cfg=cfg, rescale=rescale, img_meta=img_meta)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method, which use horizontal bboxes for NMS,
        but return the rotated bboxes result.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert with_nms, '`with_nms` must be True in RPNHead'
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = results.bboxes[:, 2], results.bboxes[:, 3]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            hbboxes = rbox2hbox(bboxes)
            _, keep_idxs = batched_nms(hbboxes, results.scores,
                                       results.level_ids, cfg.nms)
            results.bboxes = torch.cat(
                [bboxes, results.scores[:, None]], dim=1)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = results.bboxes[:, -1]
            results = results[:cfg.max_per_img]
            # TODO: This would unreasonably show the 0th class label
            #  in visualization
            results.labels = results.scores.new_zeros(
                len(results), dtype=torch.long)
            del results.level_ids
        else:
            # To avoid some potential error
            results_ = InstanceData()
            if isinstance(results.bboxes, BaseBoxes):
                results_.bboxes = results.bboxes.empty_boxes()
            else:
                results_.bboxes = results.scores.new_zeros(0, 6)
            results_.scores = results.scores.new_zeros(0)
            results_.labels = results.scores.new_zeros(0)
            results = results_
        return results

    def get_anchors(self,
                    featmap_sizes: List[tuple],
                    batch_img_metas: List[dict],
                    device: Union[torch.device, str] = 'cuda') \
            -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            batch_img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors.
                Defaults to cuda.

        Returns:
            tuple:

                - anchor_list (list[list[Tensor]]): Anchors of each image.
                - valid_flag_list (list[list[Tensor]]): Valid flags of each
                  image.
        """
        num_imgs = len(batch_img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def centerness_target(self, rboxes: Tensor) -> Tensor:
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)
        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = rboxes[:, [0, 2]]
        top_bottom = rboxes[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def anchor_center(self, anchors: RotatedBoxes) -> Tensor:
        # Get anchor centers from anchors.
        return anchors.centers

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        """Get the number of valid anchors in every level."""

        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
