# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import fvcore.nn.weight_init as weight_init
import torch
from detectron2.structures import PolygonMasks
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY, get_event_storage, mask_rcnn_loss, \
    mask_rcnn_inference


def multi_mask_rcnn_loss(pred_mask_logits, instances, n_masks_per_roi=2):
    """
    pred_mask_logits (Tensor): A tensor of shape (B, C*I, Hmask, Wmask) or (B, 1, Hmask, Wmask)
        for class-specific or class-agnostic, where B is the total number of predicted masks
        in all images, I is the number of instances per class, C is the number of foreground classes, and Hmask,
        Wmask are the height and width of the mask predictions. The values are logits.
        NOTE on ordering of second channel: pred_mask_logits[:, ::n_masks_per_roi, :, :] should give one instance mask
        for each class (masks per class should be 'grouped' together).
    instances (list[Instances]): A list of N Instances, where N is the number of images
        in the batch. These instances are in 1:1
        correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask, ...) associated with
        each instance are stored in fields.
    n_masks_per_roi: The number of masks per ROI/cls (I)
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    n_masks = pred_mask_logits.size(1)
    n_cls = n_masks // n_masks_per_roi
    assert (float(n_masks) / n_masks_per_roi) == n_cls, ValueError('Should be divisible by n_instances_per_class')

    # Temporary test: len(instances) == pred_mask_logits.size(1) / n_masks_per_roi
    logit_sets = [pred_mask_logits[:, i::n_masks_per_roi, :, :] for i in range(n_masks_per_roi)]
    # losses = [custom_mask_rcnn_loss(logits, instances, [i.gt_masks for i in instances]) for logits in
    #           logit_sets]
    
    gt_sets = [[i.gt_masks for i in instances], [i.gt_second_best_masks for i in instances]]
    losses = [custom_mask_rcnn_loss(logits, instances, gt_set) for logits, gt_set in zip(logit_sets, gt_sets)]
    return sum(losses)


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

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)

    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
    )
    return mask_loss


def multi_mask_rcnn_inference(pred_mask_logits, pred_instances, n_instances_per_class=2, inference_channel=1):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C*I, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, I is the number of instances per class, C is the number of foreground classes, and Hmask,
            Wmask are the height and width of the mask predictions. The values are logits.
            NOTE on ordering of second channel: pred_mask_logits[:, ::n_masks_per_roi, :, :] should give one instance
            mask
            for each class (masks per class should be 'grouped' together).
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    assert inference_channel in range(1, n_instances_per_class+1)

    assert pred_mask_logits.shape[1] % n_instances_per_class == 0, \
        f'{pred_mask_logits.shape[1]} % {n_instances_per_class} != 0'  # Should be C*I
    cls_agnostic_mask = pred_mask_logits.size(1) == n_instances_per_class

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices_ch1 = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred_ch1 = pred_mask_logits[indices_ch1, class_pred * n_instances_per_class][:, None].sigmoid()
        indices_ch2 = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred_ch2 = pred_mask_logits[indices_ch2, class_pred * n_instances_per_class + 1][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred_ch1 = mask_probs_pred_ch1.split(num_boxes_per_image, dim=0)
    mask_probs_pred_ch2 = mask_probs_pred_ch2.split(num_boxes_per_image, dim=0)

    for prob_ch1, prob_ch2, instances in zip(mask_probs_pred_ch1, mask_probs_pred_ch2, pred_instances):
        instances.pred_masks1 = prob_ch1  # (1, Hmask, Wmask)
        instances.pred_masks2 = prob_ch2  # (1, Hmask, Wmask)
        instances.pred_masks = instances.pred_masks1 if inference_channel == 1 else instances.pred_masks2

@ROI_MASK_HEAD_REGISTRY.register()
class CustomMaskRCNNConvUpsampleHeadAPD(nn.Module):
    """
    A custom mask head that produces more than one instance per thing class.  Similar to
    MaskRCNNConvUpsampleHead.
    """
    num_instances_per_class = 2

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(CustomMaskRCNNConvUpsampleHeadAPD, self).__init__()

        # fmt: off
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes * self.num_instances_per_class, kernel_size=1, stride=1,
                                padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)


def build_custom_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.CUSTOM_NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.CUSTOM_NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
