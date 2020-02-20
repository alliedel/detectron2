## The goal of this script is to load a pre-existing Mask R-CNN model and run it on COCO.
## Then, we will export images with the predicted bounding boxes.
## Then, we will export images with the proposal bounding boxes.
## Then, we will analyze the combination of GT and prediction boxes to see how many predictions include
# co-occurrence with another object.

## Later objectives:
# Use this on the Kitchen dataset
# Change the training loss to re-learn the instance generation.

import cv2
import os
import matplotlib.pyplot as plt
import torch
from export_proposal_helpers import FigExporter, cv2_imshow, get_maskrcnn_cfg, DETECTRON_REPO
from detectron2.engine import DefaultPredictor


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
cfg = get_maskrcnn_cfg()
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
from detectron2.engine import DefaultTrainer, DefaultPredictor

print('CFG: ')
print(cfg)
assert len(cfg.DATASETS.TRAIN) == 1
dataloaders = {
    s: DefaultTrainer.build_test_loader(cfg, {'train': cfg.DATASETS.TRAIN[0], 'val': cfg.DATASETS.TEST[0]}[s]) for s
    in ('train', 'val')
}
print('Number of training images: ', len(dataloaders['train']))
print('Number of validation images: ', len(dataloaders['val']))

# # f. Export an example
# datapoint = None
# for idx, i in enumerate(train_data_loader):
#     if idx == 0:
#         datapoint = i
#
# example_img = datapoint['image']
# data_name = os.path.splitext(os.path.basename(datapoint['file_name']))[0]
# from detectron2.structures.instances import Instances
# v = Visualizer(example_img, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2_imshow(v.get_image()[:, :, ::-1])
# exporter.export_gcf('test image')

# g. Run Mask R-CNN predictions on all COCO images

# for each image in training set

from export_proposal_helpers import just_inference_on_dataset, get_image_identifiers

outdir = os.path.join('output', os.path.splitext(os.path.basename(config_filepath))[0])
if not os.path.exists(outdir):
    os.makedirs(outdir)

for split, data_loader in dataloaders.items():
    n_points = None  # Set to 10 or so for debugging
    just_inference_on_dataset(predictor.model, data_loader, outdir, n_points)


# Write file list to path
filelists = {s: os.path.join(outdir, 'filelist_{}.txt'.format(s))
             for s in ('train', 'val')}

for s in ('val', 'train'):
    if not os.path.exists(filelists[s]): # generate filelist
        identifiers = get_image_identifiers(dataloaders[s], identifier_strings=('file_name', 'image_id'))
        list_of_files = [idnt['file_name'] for idnt in identifiers]
        with open(filelists[s], 'w') as fid:
            fid.write('\n'.join(list_of_files))
            fid.write('\n')
            print('Wrote list of {} files to {}'.format(len(list_of_files), filelists[s]))
    else:
        print('File list already exists at {}'.format(filelists[s]))
