from typing import List

import torch
from scipy import optimize
import numpy as np
from torch.nn import functional as F

from detectron2.layers import cat
from detectron2.structures import PolygonMasks
from detectron2.utils.events import get_event_storage


def solve_matching_problem(cost_tensor: torch.Tensor):
    """
    Returns matching assignment, sorted by row index.
    """
    if torch is not None:
        assert type(cost_tensor) is np.ndarray or torch.is_tensor(cost_tensor)
    else:
        assert type(cost_tensor) is np.ndarray
    cost_tensor_for_assignment = cost_tensor.detach() if cost_tensor.requires_grad else cost_tensor
    if cost_tensor_for_assignment.is_cuda:
        cost_tensor_for_assignment = cost_tensor_for_assignment.cpu()
    row_ind, col_ind = optimize.linear_sum_assignment(cost_tensor_for_assignment)
    ind_idxs_sorted_by_row = np.argsort(row_ind)
    col_ind = [col_ind[idx] for idx in ind_idxs_sorted_by_row]
    return col_ind


def get_matching_xent_losses(instances, pred_mask_logits, n_masks_per_roi):
    n_images = len(instances)
    assert n_images == 1, NotImplementedError('Only handles batch size = 1 at the moment')

    mask_side_len = pred_mask_logits.size(2)

    instances_per_image = instances[0]

    gt_1 = instances_per_image.gt_masks.crop_and_resize(
        instances_per_image.proposal_boxes.tensor, mask_side_len).to(device=pred_mask_logits.device)
    gt_2 = instances_per_image.gt_second_best_masks.crop_and_resize(
        instances_per_image.proposal_boxes.tensor, mask_side_len).to(device=pred_mask_logits.device)

    gt_pairs = [torch.stack([g1, g2]) for g1, g2 in zip(gt_1, gt_2)]

    gt_classes = instances_per_image.gt_classes.to(dtype=torch.int64)

    pred_mask_pairs = [[p[i::n_masks_per_roi, :, :][gt_class] for i in range(n_masks_per_roi)] for p, gt_class in
                       zip(pred_mask_logits, gt_classes)]

    for idx, (gt_pair, pred_pair) in enumerate(zip(gt_pairs, pred_mask_pairs)):
        xent_losses = torch.zeros((len(gt_pair), len(pred_pair)), device=pred_mask_logits.device)
        for i, pred in enumerate(pred_pair):
            for j, gt in enumerate(gt_pair):
                xent_losses[i, j] = F.binary_cross_entropy_with_logits(
                    pred, gt.to(dtype=torch.float32), reduction='mean')

    match_cols = solve_matching_problem(xent_losses)

    return xent_losses[torch.arange(xent_losses.shape[0], device=xent_losses.device), match_cols]


def maskwise_mask_rcnn_loss(pred_mask_logits, instances, gt_masks_raw: List[PolygonMasks]):
    """
    CUSTOM version of the below description (original mask_rcnn_loss function).
    If gt_masks == [i.gt_masks for i in instances], behavior is the same. We add this customization so we can give it
    secondary groundtruth (which may exist in another field, and need reassignment)

    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        gt_masks_raw : ground-truth labels for mask, of length N

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image, gt_masks_per_image in zip(instances, gt_masks_raw):
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = gt_masks_per_image.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5

    maskwise_mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks.to(dtype=torch.float32), reduce=False
    ).sum(dim=2).sum(dim=1)

    return maskwise_mask_loss


def custom_mask_rcnn_loss(pred_mask_logits, instances, gt_masks_raw: List[PolygonMasks]):
    """
    CUSTOM version of the below description (original mask_rcnn_loss function).
    If gt_masks == [i.gt_masks for i in instances], behavior is the same. We add this customization so we can give it
    secondary groundtruth (which may exist in another field, and need reassignment)

    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        gt_masks_raw : ground-truth labels for mask

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image, gt_masks_per_image in zip(instances, gt_masks_raw):
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = gt_masks_per_image.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5

    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
    )

    # Log the training accuracy (using gt classes and 0.5 threshold)
    log_accuracy(gt_masks_bool, pred_mask_logits, get_event_storage())

    return mask_loss


def log_accuracy(gt_masks_bool, pred_mask_logits, storage):
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)