# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Common data processing utilities that are used in a
typical object detection data pipeline.
"""
import imantics
import logging
import numpy as np
import torch
from detectron2.structures.masks import polygons_to_bitmask
from fvcore.common.file_io import PathManager
import os, cv2
from PIL import Image
import pycocotools.mask as mask_util

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
)

# For bitmap->polygon conversion, sometimes it's not perfect and one of the polygons is just two points (I dont
# know why).  We only save in noop.  Set to None if you don't want to save it.
SAVE_DROPPED_POLYGONS_DIR = "./scrap_local/dropped_polygons/"

from . import transforms as T
from .catalog import MetadataCatalog


class SizeMismatchError(ValueError):
    """
    When loaded image has difference width/height compared with annotation.
    """


def read_image(file_name, format=None):
    """
    Read an image into the given format.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"

    Returns:
        image (np.ndarray): an HWC image
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image


def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            raise SizeMismatchError(
                "mismatch (W,H), got {}, expect {}".format(image_wh, expected_wh)
            )


def transform_proposals(dataset_dict, image_shape, transforms, min_box_side_len, proposal_topk):
    """
    Apply transformations to the proposals in dataset_dict, if any.

    Args:
        dataset_dict (dict): a dict read from the dataset, possibly
            contains fields "proposal_boxes", "proposal_objectness_logits", "proposal_bbox_mode"
        image_shape (tuple): height, width
        transforms (TransformList):
        min_box_side_len (int): keep proposals with at least this size
        proposal_topk (int): only keep top-K scoring proposals

    The input dict is modified in-place, with abovementioned keys removed. A new
    key "proposals" will be added. Its value is an `Instances`
    object which contains the transformed proposals in its field
    "proposal_boxes" and "objectness_logits".
    """
    if "proposal_boxes" in dataset_dict:
        # Transform proposal boxes
        boxes = transforms.apply_box(
            BoxMode.convert(
                dataset_dict.pop("proposal_boxes"),
                dataset_dict.pop("proposal_bbox_mode"),
                BoxMode.XYXY_ABS,
            )
        )
        boxes = Boxes(boxes)
        objectness_logits = torch.as_tensor(
            dataset_dict.pop("proposal_objectness_logits").astype("float32")
        )

        boxes.clip(image_shape)
        keep = boxes.nonempty(threshold=min_box_side_len)
        boxes = boxes[keep]
        objectness_logits = objectness_logits[keep]

        proposals = Instances(image_shape)
        proposals.proposal_boxes = boxes[:proposal_topk]
        proposals.objectness_logits = objectness_logits[:proposal_topk]
        dataset_dict["proposals"] = proposals


def transform_instance_annotations(
        annotation, transforms, image_size, *, keypoint_hflip_indices=None,
        image_id=None
):
    """
    Apply transforms to box, segmentation and keypoints of annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
        image_id: Only used for debugging.
    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = transforms.apply_box([bbox])[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    # Old version
    # # each instance contains 1 or more polygons
    # polygons = [np.asarray(p).reshape(-1, 2) for p in annotation["segmentation"]]
    # annotation["segmentation"] = [p.reshape(-1) for p in transforms.apply_polygons(polygons)]

    # Copied from updated version of detectron2
    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]

        elif isinstance(segm, dict):
            # RLE
            im_sz = segm['size']
            H, W = im_sz
            bitmask = mask_util.decode(segm)
            # Convert to polygons
            polygons_transf = imantics.Mask(bitmask).polygons().points

            # img = np.ones((im_sz[0], im_sz[1], 1), np.uint8) * 255
            # for polygon in polygons_transf:
            #     for i in range(1, polygon.shape[0]):
            #         cv2.line(img, tuple(polygon[i - 1, :]), tuple(polygon[i, :]), color=(0, 255, 0), thickness=9)
            # cv2.imwrite('bitmask.png', bitmask.astype(np.uint8) * 255)
            # cv2.imwrite('tmp.png', img)
            #
            # imantics uses r,c; coco uses x,y

            if 0:
                xy_polygons = []
                for polygon in polygons_transf:
                    p = polygon[:, ::-1]
                    p[:, 1] = H - 1 - p[:, 1]  # Double check -1?
                    xy_polygons.append(p)
                polygons_transf = xy_polygons

            # apply_polygons expects (x, y) format.
            # (x, y) format in absolute coordinates.
            # The coordinates are not pixel indices. Coordinates on an image of
            # shape (H, W) are in range [0, W] or [0, H].
            segms = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons_transf)
            ]

            good_polygons = []
            bad_polygons = []
            # Get rid of 'bad' polygons that detectron2/COCO format won't accept.
            for poly_i, polygon in enumerate(segms):
                if not (len(polygon) % 2 == 0 and len(polygon) >= 6):
                    bad_polygons.append(polygon)
                    continue
                if len(polygon) == 8:  # old version of pycocoutils is dumb about 'bounding box'-looking things
                    new_polygon = np.ndarray((10,), dtype=polygon.dtype)
                    new_polygon[:8] = polygon
                    new_polygon[8:] = polygon[:2]
                    good_polygons.append(new_polygon)
                else:
                    good_polygons.append(polygon)

            # Debug mode
            if SAVE_DROPPED_POLYGONS_DIR is not None:
                new_sz = transforms.apply_box(np.array([0, 0, W, H])[None, :])[0]
                assert new_sz[0] == new_sz[1] == 0
                new_W, new_H = new_sz[2:]
                if not os.path.exists(SAVE_DROPPED_POLYGONS_DIR):
                    os.mkdir(SAVE_DROPPED_POLYGONS_DIR)
                if len(bad_polygons) > 0:
                    subdir = os.path.join(SAVE_DROPPED_POLYGONS_DIR, f"{image_id}")
                    if not os.path.exists(subdir):
                        os.mkdir(subdir)
                    cv2.imwrite(f"{subdir}/mask.png", bitmask * 255)
                    np.save(f"{subdir}/mask.npy", bitmask)
                    segms_l = len(segms)
                    segms = good_polygons
                    assert len(segms) < segms_l, 'For temp debug only: code is incorrect.'
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Dropped a tiny polygon from {image_id}")

                # Debug only (should never happen after we weeded out the bad ones)
                for poly_i, polygon in enumerate(segms):
                    assert len(polygon.shape) == 1
                    if not (len(polygon) % 2 == 0 and len(polygon) >= 6):
                        raise Exception

                if SAVE_DROPPED_POLYGONS_DIR is not None:
                    if len(bad_polygons) > 0:
                        segm_v3 = [s.reshape(-1, ) for s in segms]
                        bitmask_v3 = polygons_to_bitmask(segm_v3, new_H, new_W)
                        cv2.imwrite(f"{subdir}/mask_after_dropped_polygon.png", bitmask_v3 * 255)
                        np.save(f"{subdir}/mask_after_dropped_polygon.npy", bitmask_v3)
                    else:
                        if np.random.binomial(1, 0.1, size=None):
                            subdir = os.path.join(SAVE_DROPPED_POLYGONS_DIR, f"{image_id}_notdropped")
                            if not os.path.exists(subdir):
                                os.mkdir(subdir)
                            cv2.imwrite(f"{subdir}/mask.png", bitmask * 255)
                            np.save(f"{subdir}/mask.npy", bitmask)
                            segm_v3 = [s.reshape(-1, ) for s in segms]
                            bitmask_v3 = polygons_to_bitmask(segm_v3, new_H, new_W)
                            cv2.imwrite(f"{subdir}/mask_after_transformation.png", bitmask_v3 * 255)
                            np.save(f"{subdir}/mask_after_transformation.npy", bitmask_v3)

            annotation["segmentation"] = segms
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    if "keypoints" in annotation:
        keypoints = transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints

    return annotation


def transform_keypoint_annotations(keypoints, transforms, image_size, keypoint_hflip_indices=None):
    """
    Transform keypoint annotations of an image.

    Args:
        keypoints (list[float]): Nx3 float in Detectron2 Dataset format.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
    """
    # (N*3,) -> (N, 3)
    keypoints = np.asarray(keypoints, dtype="float64").reshape(-1, 3)
    keypoints[:, :2] = transforms.apply_coords(keypoints[:, :2])

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1

    # Alternative way: check if probe points was horizontally flipped.
    # probe = np.asarray([[0.0, 0.0], [image_width, 0.0]])
    # probe_aug = transforms.apply_coords(probe.copy())
    # do_hflip = np.sign(probe[1][0] - probe[0][0]) != np.sign(probe_aug[1][0] - probe_aug[0][0])  # noqa

    # If flipped, swap each keypoint with its opposite-handed equivalent
    if do_hflip:
        assert keypoint_hflip_indices is not None
        keypoints = keypoints[keypoint_hflip_indices, :]

    # Maintain COCO convention that if visibility == 0, then x, y = 0
    # TODO may need to reset visibility for cropped keypoints,
    # but it does not matter for our existing algorithms
    keypoints[keypoints[:, 2] == 0] = 0
    return keypoints


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            masks = PolygonMasks(segms)
        # Old version
        # else:
        #     assert mask_format == "bitmask", mask_format
        #     masks = BitMasks.from_polygon_masks(polygons, *image_size)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target


def annotations_to_instances_rotated(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Compared to `annotations_to_instances`, this function is for rotated boxes only

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            Containing fields "gt_boxes", "gt_classes",
            if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [obj["bbox"] for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = RotatedBoxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    return target


def filter_empty_instances(instances, by_box=True, by_mask=True):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty())
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    return instances[m]


def create_keypoint_hflip_indices(dataset_names):
    """
    Args:
        dataset_names (list[str]): list of dataset names
    Returns:
        ndarray[int]: a vector of size=#keypoints, storing the
        horizontally-flipped keypoint indices.
    """

    check_metadata_consistency("keypoint_names", dataset_names)
    check_metadata_consistency("keypoint_flip_map", dataset_names)

    meta = MetadataCatalog.get(dataset_names[0])
    names = meta.keypoint_names
    # TODO flip -> hflip
    flip_map = dict(meta.keypoint_flip_map)
    flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in flip_map else flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return np.asarray(flip_indices)


def gen_crop_transform_with_instance(crop_size, image_size, instance):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    crop_size = np.asarray(crop_size, dtype=np.int32)
    bbox = BoxMode.convert(instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
    return T.CropTransform(x0, y0, crop_size[1], crop_size[0])


def check_metadata_consistency(key, dataset_names):
    """
    Check that the datasets have consistent metadata.

    Args:
        key (str): a metadata key
        dataset_names (list[str]): a list of dataset names

    Raises:
        AttributeError: if the key does not exist in the metadata
        ValueError: if the given datasets do not have the same metadata values defined by key
    """
    if len(dataset_names) == 0:
        return
    logger = logging.getLogger(__name__)
    entries_per_dataset = [getattr(MetadataCatalog.get(d), key) for d in dataset_names]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(key, dataset_names[idx], str(entry))
            )
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(
                    key, dataset_names[0], str(entries_per_dataset[0])
                )
            )
            raise ValueError("Datasets have different metadata '{}'!".format(key))


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    logger = logging.getLogger(__name__)
    tfm_gens = []
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        tfm_gens.append(T.RandomFlip())
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens
