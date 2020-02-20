## The goal of this script is to load a pre-existing Mask R-CNN model and run it on COCO.
## Then, we will export images with the predicted bounding boxes.
## Then, we will export images with the proposal bounding boxes.
## Then, we will analyze the combination of GT and prediction boxes to see how many predictions include
# co-occurrence with another object.

import gc
import glob
import cv2
import os
import numpy as np
from PIL import Image
import sys
import psutil
## Later objectives:
# Use this on the Kitchen dataset
# Change the training loss to re-learn the instance generation.
import time
import torch
import torch.distributed
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.evaluator import inference_context
from pprint import pprint
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.modeling.postprocessing import detector_postprocess

from export_proposal_helpers import FigExporter, cv2_imshow, get_maskrcnn_cfg, DETECTRON_REPO

exporter = None


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def equal_ids(id1, id2):
    return str(id1).rstrip('0') == str(id2).rstrip('0')


def find_datapoint(dataloader, image_id):
    i = 0
    print('dataloader')
    for ds in dataloader:
        if i % 10 == 0:
            print(i)
        for d in ds:
            if equal_ids(d['image_id'], image_id):
                return d
        i += 1
    raise Exception('{} not found in dataloader'.format(image_id))


def run_vanilla_inference(predictor, inputs, train=False):
    if train is False:
        with inference_context(predictor.model), torch.no_grad():
            outputs = predictor.model(inputs)
    else:
        outputs = predictor.model(inputs)
    return outputs


def run_inference(predictor, inputs):
    with inference_context(predictor.model), torch.no_grad():
        # Get proposals
        images = predictor.model.preprocess_image(inputs)
        features = predictor.model.backbone(images.tensor)
        proposalss, proposal_lossess = predictor.model.proposal_generator(images, features, None)

        # Get instance boxes, masks, and proposal idxs
        outputs, extra_proposal_details = predictor.model(inputs, trace_proposals=True)

        return {'outputs': outputs,
                'proposalss': proposalss,
                'proposal_lossess': proposal_lossess,
                'extra_proposal_details': extra_proposal_details
                }


def convert_datapoint_to_image_format(img, out_shape, cfg):
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    if cfg.INPUT.FORMAT == "BGR":
        img = np.asarray(img[:, :, [2, 1, 0]])
    else:
        img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))
    if cfg.INPUT.FORMAT == "BGR":
        img = np.asarray(img[:, :, [2, 1, 0]])
    else:
        img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))
    img = cv2.resize(img, out_shape[::-1])
    return img


def build_dataloader(cfg):
    dataloaders_eval = {
        'val': DefaultTrainer.build_test_loader(cfg, {'train': cfg.DATASETS.TRAIN[0], 'val': cfg.DATASETS.TEST[0]}[s])
        for s
        in ('train', 'val')
    }
    train_dataloader = DefaultTrainer.build_train_loader(cfg)
    return train_dataloader


def main(config_filepath=f"{DETECTRON_REPO}/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
         image_id='486536', flip_lr=False):
    n_existing_exporter_images = len(exporter.generated_figures)
    saved_input_file = '{}_{}.pt'.format(os.path.splitext(os.path.basename(__file__))[0], image_id)
    cfg = get_maskrcnn_cfg(config_filepath)
    predictor = DefaultPredictor(cfg)
    # predictor.model.training = True
    if not os.path.exists(saved_input_file):
        print('Saved input file did not exist.  Creating now.: {}'.format(saved_input_file))
        dataloader = build_dataloader(cfg)
        datapoint = find_datapoint(dataloader, image_id)
        torch.save(datapoint, saved_input_file)
        gc.collect()
        del dataloader
    datapoint = torch.load(saved_input_file)
    if type(datapoint) is not list:
        datapoint = [datapoint]
    if flip_lr:
        assert all(d['image'].shape[0] == 3 for d in datapoint)
        for d in datapoint:
            d['image'] = d['image'].flip(dims=(2,))
    image_filenames = [d['file_name'] for d in datapoint]
    input_images = [d['image'] for d in datapoint]
    input_images = [np.asarray(img.permute(1, 2, 0)[:, :, [2, 1, 0]]) for img in input_images]
    input_images_from_files = [cv2.imread(fn) for fn in image_filenames]
    input_images = [convert_datapoint_to_image_format(im, im2.shape[:2], cfg)
                    for im, im2 in zip(input_images, input_images_from_files)]

    output = run_vanilla_inference(predictor, datapoint, train=False)
    outputs_d = run_inference(predictor, datapoint)


if __name__ == '__main__':
    exporter = FigExporter()
    image_ids = ['486536', '306284', '9']
    for image_id in image_ids:
        for flip_lr in [False, True]:
            main(image_id=image_id, flip_lr=flip_lr)
            gc.collect()
