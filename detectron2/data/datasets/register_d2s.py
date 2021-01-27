from detectron2.data.datasets import register_coco_instances


def register_all_d2s():
    register_coco_instances("d2s_train", {}, "data/datasets/d2s/annotations/D2S_training.json",
                            "data/datasets/d2s/images/")
    register_coco_instances("d2s_val", {}, "data/datasets/d2s/annotations/D2S_validation.json",
                            "data/datasets/d2s/images/")
    register_coco_instances("d2s_val_clutter", {},
                            "data/datasets/d2s/annotations/D2S_validation_clutter.json",
                            "data/datasets/d2s/images/")
    register_coco_instances("d2s_val_occlusion", {},
                            "data/datasets/d2s/annotations/D2S_validation_occlusion.json",
                            "data/datasets/d2s/images/")
    register_coco_instances("d2s_val_wo_occlusion", {},
                            "data/datasets/d2s/annotations/D2S_validation_wo_occlusion.json",
                            "data/datasets/d2s/images/")


register_all_d2s()


# register_all_coco()

# # Use Custom Datasets
#
# If you want to use a custom dataset while also reusing detectron2's data loaders,
# you will need to
#
# 1. Register your dataset (i.e., tell detectron2 how to obtain your dataset).
# 2. Optionally, register metadata for your dataset.
#
# Next, we explain the above two concepts in details.
#
# The [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
# has a working example of how to register and train on a dataset of custom formats.
#
#
# ### Register a Dataset
#
# To let detectron2 know how to obtain a dataset named "my_dataset", you will implement
# a function that returns the items in your dataset and then tell detectron2 about this
# function:
# ```python
# def get_dicts():
#   ...
#   return list[dict] in the following format
#
# from detectron2.data import DatasetCatalog
# DatasetCatalog.register("my_dataset", get_dicts)
# ```
#
# Here, the snippet associates a dataset "my_dataset" with a function that returns the data.
# If you do not modify downstream code (i.e., you use the standard data loader and data mapper),
# then the function has to return a list of dicts in detectron2's standard dataset format, described
# next.
#
# For standard tasks
# (instance detection, instance/semantic/panoptic segmentation, keypoint detection),
# we use a format similar to COCO's json annotations
# as the basic dataset representation.
#
# The format uses one dict to represent the annotations of
# one image. The dict may have the following fields.
# The fields are often optional, and some functions may be able to
# infer certain fields from others if needed, e.g., the data loader
# can load an image from "file_name" if the "image" field is not available.
#
# + `file_name`: the full path to the image file.
# + `sem_seg_file_name`: the full path to the ground truth semantic segmentation file.
# + `image`: the image as a numpy array.
# + `sem_seg`: semantic segmentation ground truth in a 2D numpy array. Values in the array represent
#    category labels.
# + `height`, `width`: integer. The shape of image.
# + `image_id` (str): a string to identify this image. Mainly used during evaluation to identify the
#   image. Each dataset may use it for different purposes.
# + `annotations` (list[dict]): the per-instance annotations of every
#   instance in this image. Each annotation dict may contain:
#   + `bbox` (list[float]): list of 4 numbers representing the bounding box of the instance.
#   + `bbox_mode` (int): the format of bbox.
#     It must be a member of
#     [structures.BoxMode](../modules/structures.html#detectron2.structures.BoxMode).
#     Currently supports: `BoxMode.XYXY_ABS`, `BoxMode.XYWH_ABS`.
#   + `category_id` (int): an integer in the range [0, num_categories) representing the category label.
#     The value num_categories is reserved to represent the "background" category, if applicable.
#   + `segmentation` (list[list[float]] or dict):
#     + If `list[list[float]]`, it represents a list of polygons, one for each connected component
#       of the object. Each `list[float]` is one simple polygon in the format of `[x1, y1, ..., xn, yn]`.
#       The Xs and Ys are either relative coordinates in [0, 1], or absolute coordinates,
#       depend on whether "bbox_mode" is relative.
#     + If `dict`, it represents the per-pixel segmentation mask in COCO's RLE format.
#   + `keypoints` (list[float]): in the format of [x1, y1, v1,..., xn, yn, vn].
#     v[i] means the visibility of this keypoint.
#     `n` must be equal to the number of keypoint categories.
#     The Xs and Ys are either relative coordinates in [0, 1], or absolute coordinates,
#     depend on whether "bbox_mode" is relative.
#
#     Note that the coordinate annotations in COCO format are integers in range [0, H-1 or W-1].
#     By default, detectron2 adds 0.5 to absolute keypoint coordinates to convert them from discrete
#     pixel indices to floating point coordinates.
#   + `iscrowd`: 0 or 1. Whether this instance is labeled as COCO's "crowd region".
# + `proposal_boxes` (array): 2D numpy array with shape (K, 4) representing K precomputed proposal boxes for this image.
# + `proposal_objectness_logits` (array): numpy array with shape (K, ), which corresponds to the objectness
#   logits of proposals in 'proposal_boxes'.
# + `proposal_bbox_mode` (int): the format of the precomputed proposal bbox.
#   It must be a member of
#   [structures.BoxMode](../modules/structures.html#detectron2.structures.BoxMode).
#   Default format is `BoxMode.XYXY_ABS`.
#
#
# If your dataset is already in the COCO format, you can simply register it by
# ```python
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
# ```
# which will take care of everything (including metadata) for you.
#
# If your dataset is in COCO format with custom per-instance annotations,
# the [load_coco_json](../modules/data.html#detectron2.data.datasets.load_coco_json) function can be used.


### "Metadata" for Datasets
#
# Each dataset is associated with some metadata, accessible through
# `MetadataCatalog.get(dataset_name).some_metadata`.
# Metadata is a key-value mapping that contains primitive information that helps interpret what's in the dataset, e.g.,
# names of classes, colors of classes, root of files, etc.
# This information will be useful for augmentation, evaluation, visualization, logging, etc.
# The structure of metadata depends on the what is needed from the corresponding downstream code.
#
#
# If you register a new dataset through `DatasetCatalog.register`,
# you may also want to add its corresponding metadata through
# `MetadataCatalog.get(dataset_name).set(name, value)`, to enable any features that need metadata.
# You can do it like this (using the metadata field "thing_classes" as an example):
#
# ```python
# from detectron2.data import MetadataCatalog
# MetadataCatalog.get("my_dataset").thing_classes = ["person", "dog"]
# ```
#
# Here is a list of metadata keys that are used by builtin features in detectron2.
# If you add your own dataset without these metadata, some features may be
# unavailable to you:
#
# * `thing_classes` (list[str]): Used by all instance detection/segmentation tasks.
#   A list of names for each instance/thing category.
#   If you load a COCO format dataset, it will be automatically set by the function `load_coco_json`.
#
# * `stuff_classes` (list[str]): Used by semantic and panoptic segmentation tasks.
#   A list of names for each stuff category.
#
# * `stuff_colors` (list[tuple(r, g, b)]): Pre-defined color (in [0, 255]) for each stuff category.
#   Used for visualization. If not given, random colors are used.
#
# * `keypoint_names` (list[str]): Used by keypoint localization. A list of names for each keypoint.
#
# * `keypoint_flip_map` (list[tuple[str]]): Used by the keypoint localization task. A list of pairs of names,
#   where each pair are the two keypoints that should be flipped if the image is
#   flipped during augmentation.
# * `keypoint_connection_rules`: list[tuple(str, str, (r, g, b))]. Each tuple specifies a pair of keypoints
#   that are connected and the color to use for the line between them when visualized.
#
# Some additional metadata that are specific to the evaluation of certain datasets (e.g. COCO):
#
# * `thing_dataset_id_to_contiguous_id` (dict[int->int]): Used by all instance detection/segmentation tasks in the
# COCO format.
#   A mapping from instance class ids in the dataset to contiguous ids in range [0, #class).
#   Will be automatically set by the function `load_coco_json`.
#
# * `stuff_dataset_id_to_contiguous_id` (dict[int->int]): Used when generating prediction json files for
#   semantic/panoptic segmentation.
#   A mapping from semantic segmentation class ids in the dataset
#   to contiguous ids in [0, num_categories). It is useful for evaluation only.
#
# * `json_file`: The COCO annotation json file. Used by COCO evaluation for COCO-format datasets.
# * `panoptic_root`, `panoptic_json`: Used by panoptic evaluation.
# * `evaluator_type`: Used by the builtin main training script to select
#    evaluator. No need to use it if you write your own main script.
#    You can just provide the [DatasetEvaluator](../modules/evaluation.html#detectron2.evaluation.DatasetEvaluator)
#    for your dataset directly in your main script.
