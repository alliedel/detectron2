## The goal of this script is to load a pre-existing Mask R-CNN model and run it on COCO.
## Then, we will export images with the predicted bounding boxes.
## Then, we will export images with the proposal bounding boxes.
## Then, we will analyze the combination of GT and prediction boxes to see how many predictions include
# co-occurrence with another object.

import cv2
import os
## Later objectives:
# Use this on the Kitchen dataset
# Change the training loss to re-learn the instance generation.
import time
import torch
import torch.distributed
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.evaluator import inference_context

from export_proposal_helpers import FigExporter, cv2_imshow, get_maskrcnn_cfg, DETECTRON_REPO


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


# I. Load pre-existing Mask R-CNN model
# a. Get example image to work with

im = cv2.imread("./input.jpg")
cv2_imshow(im)
exporter = FigExporter()
exporter.export_gcf('input')

# b. Load Mask R-CNN model
config_filepath = f"{DETECTRON_REPO}/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
cfg = get_maskrcnn_cfg(config_filepath)
predictor = DefaultPredictor(cfg)

# c. Get Mask R-CNN predictions and print boxes
outputs = predictor(im)

from pprint import pprint
pprint(outputs["instances"].pred_classes)
pprint(outputs["instances"].pred_boxes)

# d. Visualize and export Mask R-CNN predictions
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(v.get_image()[:, :, ::-1])
exporter.export_gcf('prediction')

# e. Load all COCO images
from detectron2.engine import DefaultTrainer

print('CFG: ')
print(cfg)
assert len(cfg.DATASETS.TRAIN) == 1
dataloaders = {
    s: DefaultTrainer.build_test_loader(cfg, {'train': cfg.DATASETS.TRAIN[0], 'val': cfg.DATASETS.TEST[0]}[s]) for s
    in ('train', 'val')
}
print('Number of training images: ', len(dataloaders['train']))
print('Number of validation images: ', len(dataloaders['val']))

outdir = os.path.join('output', os.path.splitext(os.path.basename(config_filepath))[0])
if not os.path.exists(outdir):
    os.makedirs(outdir)
train = True
# for split, data_loader in dataloaders.items():
split, data_loader = 'train', dataloaders['train']
if 1:
    n_points = None  # Set to 10 or so for debugging

    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    total = len(data_loader)  # inference data loader must have a fixed length

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    n_points = 4
    if train is False:
        with inference_context(predictor.model), torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                if idx >= n_points:
                    break
                outputs = predictor.model(inputs)
    else:
        for idx, inputs in enumerate(data_loader):
            if idx >= n_points:
                break
            outputs = predictor.model(inputs)
