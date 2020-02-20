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


def dictoflists_to_listofdicts(dictoflists):
    n = None
    for k, v in dictoflists.items():
        if n is None:
            n = len(v)
        else:
            assert len(v) == n
    listofdicts = [{k: v[i] for k, v in dictoflists.items()} for i in range(n)]
    return listofdicts


def run_vanilla_evaluation(images, cfg, outputs, image_ids, model=None):
    for img, output, image_id in zip(images, outputs, image_ids):
        if img.shape[2] == 3:
            output_size = img.shape[:2]
        else:
            output_size = img.shape[1:]
        if output['instances'].image_size != output_size:
            # for some reason, it wants an extra dimension...
            B, H, W = output['instances'].pred_masks.shape
            output['instances'].pred_masks = output['instances'].pred_masks.resize(B, 1, H, W)
            # output['instances'] = detector_postprocess(output['instances'], output_size[0], output_size[1])
        # d. Visualize and export Mask R-CNN predictions
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        proposal_score_thresh = None if model is None else model.roi_heads.test_score_thresh
        visualize(img, metadata, instances=output, proposals=None, image_id=str(image_id),
                  extra_proposal_details=None,
                  scale=2.0, proposal_score_thresh=proposal_score_thresh)


def run_evaluation(images, cfg, outputs_d, image_ids, model=None):
    outputs = outputs_d.pop('outputs')
    proposalss = outputs_d.pop('proposalss')
    extra_proposal_detailss = outputs_d.pop('extra_proposal_details', None)
    if type(extra_proposal_detailss) is dict:
        extra_proposal_detailss = dictoflists_to_listofdicts(extra_proposal_detailss)

    for img, output, proposals, extra_proposal_details, image_id in \
            zip(images, outputs, proposalss, extra_proposal_detailss, image_ids):
        if img.shape[2] == 3:
            output_size = img.shape[:2]
        else:
            output_size = img.shape[1:]
        if proposals.image_size != output_size:
            proposals = detector_postprocess(proposals, output_size[0], output_size[1])
        if output['instances'].image_size != output_size:
            # for some reason, it wants an extra dimension...
            B, H, W = output['instances'].pred_masks.shape
            output['instances'].pred_masks = output['instances'].pred_masks.resize(B, 1, H, W)
            output['instances'] = detector_postprocess(output['instances'], output_size[0], output_size[1])
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        if cfg.INPUT.FORMAT == "BGR":
            img = np.asarray(img[:, :, [2, 1, 0]])
        else:
            img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))
        # d. Visualize and export Mask R-CNN predictions
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        proposal_score_thresh = None if model is None else model.roi_heads.test_score_thresh
        visualize(img, metadata, instances=output, proposals=proposals, image_id=str(image_id),
                  extra_proposal_details=extra_proposal_details,
                  scale=2.0, proposal_score_thresh=proposal_score_thresh)


def visualize(img, metadata, instances, proposals, image_id, extra_proposal_details=None,
              scale=2.0, map_instance_to_proposal_vis=True, proposal_score_thresh=None):
    if img is not None:
        cv2_imshow(img[:, :, ::-1].astype('uint8'))
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_input')

    if proposals is not None:
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
        v._default_font_size = v._default_font_size * 1.5
        proposals.pred_boxes = proposals.proposal_boxes
        v = v.draw_instance_predictions(proposals.to('cpu'))
        cv2_imshow(v.get_image()[:, :, ::-1])
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_proposals')

    if extra_proposal_details is not None:
        selected_proposal_idxs = extra_proposal_details['selected_proposal_idxs']
    else:
        selected_proposal_idxs = None

    if extra_proposal_details is not None:
        extra_proposal_details['scores'] = extra_proposal_details['scores'].to('cpu')
        proposal_subset = (extra_proposal_details['scores'][:, :-1] > proposal_score_thresh).nonzero()
        proposal_subset_inds = proposal_subset[:, 0]
        proposal_subset_classes = proposal_subset[:, 1]
        proposals_past_thresh = proposals[proposal_subset_inds]
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
        v._default_font_size = v._default_font_size * 1.5
        proposals_past_thresh.pred_boxes = proposals_past_thresh.proposal_boxes
        if map_instance_to_proposal_vis:
            proposals_past_thresh.scores = extra_proposal_details['scores'][
                proposal_subset[:, 0], proposal_subset[:, 1]]
            proposals_past_thresh.pred_classes = proposal_subset_classes

        proposals_past_thresh = proposals_past_thresh.to('cpu')
        v = v.draw_instance_predictions(proposals_past_thresh)
        cv2_imshow(v.get_image()[:, :, ::-1])
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + 'proposals_past_thresh')

    if selected_proposal_idxs is not None:
        assert len(selected_proposal_idxs) == len(instances['instances'])
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
        v._default_font_size = v._default_font_size * 1.5
        proposals_selected = proposals[selected_proposal_idxs]
        proposals_selected.pred_boxes = proposals_selected.proposal_boxes
        if map_instance_to_proposal_vis:
            proposals_selected.scores = instances['instances'].scores
            proposals_selected.pred_classes = instances['instances'].pred_classes

        v = v.draw_instance_predictions(proposals_selected.to('cpu'))
        cv2_imshow(v.get_image()[:, :, ::-1])
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_selected_proposals')

    if instances is not None:
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
        v._default_font_size = v._default_font_size * 1.5
        v = v.draw_instance_predictions(instances["instances"].to("cpu"))
        cv2_imshow(v.get_image()[:, :, ::-1])
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_prediction')


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
    run_vanilla_evaluation(input_images, cfg, output, image_ids=[str(d['image_id']) + ('_flip' if flip_lr else '')
                                                                 for d in datapoint], model=predictor.model)

    n_existing_exporter_images = len(exporter.generated_figures)
    outputs_d = run_inference(predictor, datapoint)
    run_evaluation(input_images, cfg, outputs_d,
                   image_ids=[str(d['image_id']) + ('_flip' if flip_lr else '') for d in datapoint],
                   model=predictor.model)
    my_image_ids = [str(d['image_id']) + ('_flip' if flip_lr else '') for d in datapoint]
    for my_image_id in my_image_ids:
        figure_name = os.path.splitext(os.path.basename(__file__))[0] + '_' + my_image_id + '_collated'
        collate_figures(exporter.generated_figures[n_existing_exporter_images:], figure_name)


def collate_figures(figures, figure_name):
    out_fig = cv2.hconcat([cv2.imread(f) for f in figures])
    print(out_fig.shape)
    fname = os.path.join(exporter.workspace_dir, figure_name + '.png')
    cv2.imwrite(fname, out_fig)
    cv2_imshow(out_fig.astype('uint8'))
    exporter.export_gcf(figure_name)


if __name__ == '__main__':
    exporter = FigExporter()
    image_ids = ['486536', '306284', '9']
    for image_id in image_ids:
        for flip_lr in [False, True]:
            main(image_id=image_id, flip_lr=flip_lr)
            gc.collect()
