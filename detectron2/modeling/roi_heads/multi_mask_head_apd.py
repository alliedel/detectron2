# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import fvcore.nn.weight_init as weight_init
import torch

from detectron2.modeling.roi_heads.mask_loss_apd import custom_mask_rcnn_loss, get_matching_xent_losses
from detectron2.structures import PolygonMasks
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY


def multi_mask_rcnn_loss(pred_mask_logits, instances, n_masks_per_roi=2, matching=True):
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

    torch.save({'pred_mask_logits': pred_mask_logits, 'instances': instances, 'n_masks_per_roi': n_masks_per_roi},
               'tmp_matching_loss.pt')

    # gt_sets[<primary>][<imgidx>0] where 0=<primary>,1=<secondary>, and batchsize=1 would mean imgidx can only be 0

    logit_sets = [pred_mask_logits[:, i::n_masks_per_roi, :, :] for i in range(n_masks_per_roi)]
    if matching is False:
        gt_sets = [[i.gt_masks for i in instances], [i.gt_second_best_masks for i in instances]]
        losses = [custom_mask_rcnn_loss(logits, instances, gt_set)
                  for logits, gt_set in zip(logit_sets, gt_sets)]
        return sum(losses)
    else:
        losses = get_matching_xent_losses(instances, pred_mask_logits, n_masks_per_roi)
    return sum(losses)


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
    assert inference_channel in range(1, n_instances_per_class + 1)

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
