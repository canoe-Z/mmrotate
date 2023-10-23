import torch
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead
from mmdet.structures.bbox import get_box_tensor
from mmengine.config import ConfigDict
from torch import Tensor
from typing import Optional

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import rbbox_overlaps
from mmrotate.structures.bbox.rotated_boxes import RotatedBoxes


@MODELS.register_module()
class ConvFCBBoxARLHead(ConvFCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def _get_targets_single(self, pos_priors: Tensor, neg_priors: Tensor,
                            pos_gt_bboxes: Tensor, pos_gt_labels: Tensor,
                            cfg: ConfigDict) -> tuple:
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_priors (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_priors (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_priors.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        reg_dim = pos_gt_bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        label_weights = pos_priors.new_zeros(num_samples)
        bbox_targets = pos_priors.new_zeros(num_samples, reg_dim + 1)
        bbox_weights = pos_priors.new_zeros(num_samples, reg_dim)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            pos_bboxes_score = pos_priors[:, -1]
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    RotatedBoxes(pos_priors[:, :-1]), pos_gt_bboxes)
                pos_bbox_targets = torch.cat(
                    [pos_bbox_targets, pos_bboxes_score[:, None]], dim=1)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = get_box_tensor(pos_gt_bboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)

        # loss_bbox
        bg_class_ind = self.num_classes
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        # do not perform bounding box regression for BG anymore.
        if pos_inds.any():
            if self.reg_decoded_bbox:
                # When the regression loss (e.g. `IouLoss`,
                # `GIouLoss`, `DIouLoss`) is applied directly on
                # the decoded bounding boxes, it decodes the
                # already encoded coordinates to absolute format.
                bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                bbox_pred = get_box_tensor(bbox_pred)
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
            else:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    5)[pos_inds.type(torch.bool),
                        labels[pos_inds.type(torch.bool)]]
            losses_bbox = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds.type(torch.bool), :-1],
                bbox_weights[pos_inds.type(torch.bool)],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        else:
            losses_bbox = bbox_pred[pos_inds].sum()

        # loss_cls
        if cls_score.numel() > 0:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            weight = torch.ones_like(labels).float()
            joint_weight = None
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                pos_decode_bbox_pred = self.bbox_coder.decode(
                    rois[pos_inds.type(torch.bool), 1:], pos_bbox_pred)
                pos_decode_bbox_target = self.bbox_coder.decode(
                    rois[pos_inds.type(torch.bool), 1:], bbox_targets[pos_inds.type(torch.bool), :-1])
                iou_targets_ini = rbbox_overlaps(
                    pos_decode_bbox_pred.tensor.detach(),
                    pos_decode_bbox_target.tensor.detach(),
                    is_aligned=True).clamp(min=1e-6).view(-1)
                pos_ious = iou_targets_ini.clone().detach()

                def normalize(x):
                    EPS = 1e-6
                    t1 = x.min()
                    t2 = min(1., x.max())
                    y = (x - t1 + EPS) / (t2 - t1 + EPS)
                    return y
                joint_weight = (
                    pos_ious * normalize(bbox_targets[pos_inds, -1])).pow(0.5)

            loss_cls_ = self.loss_cls(
                cls_score,
                labels,
                joint_weight,
                weight=weight,
                avg_factor=avg_factor,
                reduction_override=reduction_override)

            if isinstance(loss_cls_, dict):
                losses.update(loss_cls_)
            else:
                losses['loss_cls'] = loss_cls_
            losses['acc'] = accuracy(cls_score, labels)
            fg_mask = labels != self.num_classes
            losses['fg_acc'] = accuracy(
                cls_score[fg_mask], labels[fg_mask])
        else:
            losses['loss_cls'] = cls_score.sum() * 0
        losses['loss_bbox'] = losses_bbox
        return losses


@MODELS.register_module()
class Shared2FCBBoxARLHead(ConvFCBBoxARLHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxARLHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@MODELS.register_module()
class Shared4Conv1FCBBoxARLHead(ConvFCBBoxARLHead):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxARLHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
