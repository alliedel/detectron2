# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.modeling.roi_heads import multi_mask_head_apd
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, select_foreground_proposals
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from .mask_head import build_mask_head, mask_rcnn_inference, mask_rcnn_loss
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals

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
        self.mask_heads = {}
        for head_type_name in MASK_HEAD_TYPES:
            if head_type_name == 'standard':
                self.mask_heads[head_type_name] = build_mask_head(
                    cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution))
            elif head_type_name == 'custom':
                self.mask_heads[head_type_name] = multi_mask_head_apd.build_custom_mask_head(
                    cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution))
            else:
                raise ValueError(f'{head_type_name} is not one of head types')

        # NOTE(allie): 'hacky' addition: Point to them directly for correct initialization (to get their weights on the
        # same CUDA device -- NOTE: This worked! (and will fail without this, as long as they are only in dictionaries)
        self.standard_mask_head = self.mask_heads[MASK_HEAD_TYPES['standard']]
        self.custom_mask_head = self.mask_heads[MASK_HEAD_TYPES['custom']]
        self.mask_head = None  # To make sure we don't use the base class instantiation accidentally.
        # TODO(Allie): mask_head=None is very sloppy.  Probably should not inherit, and should just use class methods,
        #  but ensures we reuse as much of detectron2's original code as possible.

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets, get_secondary_matches=None):
        """
        ** APD modified from StandardROIHeads in roi_heads.py to handle matching of more than one groundtruth 
        Same functionality as its parent counterpart, but assigns *all* groundtruth overlapping with the box of the
        same class to the box.
        **

        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        get_secondary_matches = get_secondary_matches if get_secondary_matches is not None \
            else self.active_mask_head == 'custom'
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            # contains indices, iou match values

            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            if get_secondary_matches:
                # APD: Multi-instance addition. Add secondary gt for foreground proposal samples
                # torch.topk:
                # topk.indices[r, c] contains the groundtruth index of the r'th best match to proposal c
                # topk.values[r, c] contains the iou overlap between proposal c and groundtruth topk.indices[r, c]
                # topk.values should be in the range (0, 1)

                # We are going to modify match_quality_marix to make values of 0 if the classes do not match.  We will then
                # get the indices of all nonzero elements.
                secondary_match_quality_matrix = match_quality_matrix
                del match_quality_matrix
                NOT_SAME_CLASS = PRIMARY_ASSIGNMENT = 0
                for gt_idx, gt_cls in enumerate(targets_per_image.gt_classes):
                    proposals_matched_to_this_gt = matched_idxs == gt_idx
                    if torch.any(proposals_matched_to_this_gt):
                        gt_of_other_clss = targets_per_image.gt_classes != gt_cls
                        # secondary_match_quality_matrix[gt_of_other_clss, proposals_matched_to_this_gt] = NOT_SAME_CLASS
                        # secondary_match_quality_matrix[gt_of_other_clss][:, proposals_matched_to_this_gt] = NOT_SAME_CLASS
                        secondary_match_quality_matrix[gt_of_other_clss.nonzero(), proposals_matched_to_this_gt] = \
                            NOT_SAME_CLASS
                        secondary_match_quality_matrix[gt_idx, proposals_matched_to_this_gt] = PRIMARY_ASSIGNMENT

                # r,c of proposal, alternate gt_idx mapping.  For instance:
                #   (0,0), (1,1), (1,3) means proposal 0 has alternate gt 0, proposal 1 has alternate gt 1, and proposal 3
                #   has alternate gt 1.
                # secondary_assignments = secondary_match_quality_matrix.nonzero()  # useful for tertiary, etc. assignments
                second_best_assignments = secondary_match_quality_matrix.max(axis=0)

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                existing_field_names = list(proposals_per_image.get_fields().keys())  # apd: use later for secondary
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])

                if get_secondary_matches:
                    # Assigning second best indices
                    sampled_second_best_targets = second_best_assignments.indices[sampled_idxs]
                    sampled_second_best_values = second_best_assignments.values[sampled_idxs]

                    primary_and_secondary_gt_classes_match = torch.all(
                        (targets_per_image.gt_classes[sampled_second_best_targets]
                         == targets_per_image.gt_classes[sampled_targets])
                        | (sampled_second_best_values == 0))
                    assert primary_and_secondary_gt_classes_match
                    no_second_best_target = (second_best_assignments.values[sampled_idxs] == 0).to(dtype=torch.bool)
                    for (trg_name, trg_value) in targets_per_image.get_fields().items():
                        trg_secondary_name = trg_name.replace('gt_', 'gt_second_best_')
                        if trg_name.startswith("gt_") and trg_name not in existing_field_names:
                            secondbest_value = trg_value[sampled_second_best_targets]
                            if trg_name == 'gt_boxes':
                                secondbest_value.tensor[no_second_best_target] = 0
                            elif trg_name == 'gt_masks':
                                assert len(no_second_best_target) == len(secondbest_value.polygons)
                                for i in no_second_best_target.nonzero():
                                    for j in range(len(secondbest_value.polygons[i])):
                                        secondbest_value.polygons[i][j][:] = 0
                            else:
                                raise NotImplementedError
                            proposals_per_image.set(trg_secondary_name, secondbest_value)

            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

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
            if self.active_mask_head is 'standard':
                return {"loss_mask": mask_rcnn_loss(mask_logits, proposals)}
            elif self.active_mask_head is 'custom':
                return {"loss_mask": multi_mask_head_apd.multi_mask_rcnn_loss(mask_logits, proposals)}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            mask_logits = self.mask_heads[self.active_mask_head](mask_features)
            if self.active_mask_head is 'standard':
                mask_rcnn_inference(mask_logits, instances)
            elif self.active_mask_head is 'custom':
                if 1:
                    mask_rcnn_inference(mask_logits, instances)
                    for inst in instances:
                        inst.pred_masks_standard = inst.pred_masks
                        inst.pred_masks = [None for _ in inst.pred_masks]
                multi_mask_head_apd.multi_mask_rcnn_inference(mask_logits, instances,
                                                              self.mask_heads[
                                                                  self.active_mask_head].num_instances_per_class)
            else:
                raise ValueError

            return instances
