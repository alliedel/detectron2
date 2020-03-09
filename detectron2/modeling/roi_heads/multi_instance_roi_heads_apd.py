# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch

from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, select_foreground_proposals, get_event_storage
from detectron2.structures import Instances
from detectron2.layers import cat
from torch.nn import functional as F
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.modeling.roi_heads.mask_head import build_mask_head, mask_rcnn_inference, mask_rcnn_loss
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY

logger = logging.getLogger(__name__)


MASK_HEAD_TYPES = {
    'custom': 'custom',
    'standard': 'standard'
}


@ROI_HEADS_REGISTRY.register()
class MultiROIHeadsAPD(StandardROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        init_mask_head = cfg.MODEL.ROI_MASK_HEAD.INIT_ACTIVATED_MASK_HEAD
        assert init_mask_head in MASK_HEAD_TYPES
        self.active_mask_head = init_mask_head
        self._init_box_head(cfg)
        self._init_mask_heads(cfg)
        self._init_keypoint_head(cfg)

    def _init_mask_heads(self, cfg):
        # fmt: off
        self.proposal_selection_function_for_loss = largest_box_per_class
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_heads = {
            head_type_name: build_mask_head(
                cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
            ) for head_type_name in MASK_HEAD_TYPES
        }

        # 'hacky' addition: Point to them directly for correct initialization (to get their weights on the same CUDA
        # device -- NOTE: This worked! (and will fail without this, as long as they are only in dictionaries)
        self.standard_mask_head = self.mask_heads[MASK_HEAD_TYPES['standard']]
        self.custom_mask_head = self.mask_heads[MASK_HEAD_TYPES['custom']]
        self.mask_head = None  # To make sure we don't use the base class instantiation accidentally.
        # TODO(Allie): mask_head=None is very sloppy.  Probably should not inherit, and should just use class methods,
        #  but ensures we reuse as much of detectron2's original code as possible.

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            mask_logits = self.mask_heads[self.active_mask_head](mask_features)
            if self.proposal_selection_function_for_loss is None:
                return {"loss_mask": mask_rcnn_loss(mask_logits, proposals)}
            else:
                return {"loss_mask": mask_rcnn_loss_with_proposal_subset(mask_logits, proposals,
                                                                         self.proposal_selection_function_for_loss)}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            mask_logits = self.mask_heads[self.active_mask_head](mask_features)
            mask_rcnn_inference(mask_logits, instances)
            return instances


def find_largest_proposal_per_image(mask_logits, proposals):
    raise NotImplementedError


def largest_box_per_class(mask_logits, proposals):
    chosen_box_idxs_batch = []
    for p in proposals:
        cbi = []
        areas = p.gt_boxes.area()
        classes = p.gt_classes
        unique_classes = p.gt_classes.unique()
        assert len(classes.shape) == 1
        for uc in unique_classes:
            mask = classes == uc
            subset_idx = torch.argmin(areas[mask])
            if subset_idx.numel() > 1:
                subset_idx = subset_idx[0]
            box_idx = torch.arange(classes.shape[0])[mask][subset_idx.item()].item()
            assert box_idx < len(p)
            cbi.append(box_idx)

        chosen_box_idxs_batch.append(cbi)
    return chosen_box_idxs_batch


def mask_rcnn_loss_with_proposal_subset(mask_logits, proposals, proposal_selection_function):
    if proposal_selection_function is None:
        selected_mask_logits, selected_proposals = mask_logits, proposals
    else:
        selected_idxs = proposal_selection_function(mask_logits, proposals)
        selected_idxs_as_one_for_logits = []
        offset = 0
        for p, si in zip(proposals, selected_idxs):
            selected_idxs_as_one_for_logits.extend([s + offset for s in si])
            offset += len(p)
        selected_mask_logits = mask_logits[selected_idxs_as_one_for_logits, :, :, :]
        selected_proposals = [p[si] for p, si in zip(proposals, selected_idxs)]

    return mask_rcnn_loss(selected_mask_logits, selected_proposals)
