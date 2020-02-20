import os

from export_proposal_helpers import get_cfg, DETECTRON_REPO, cv2_imshow, FigExporter


def outfile_from_filename(filename):
    idtf = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(outdir, 'output_{}.pt'.format(idtf))


cfg = get_cfg()
config_filepath = f"{DETECTRON_REPO}/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
cfg.merge_from_file(config_filepath)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
predictor = DefaultPredictor(cfg)
import ipdb; ipdb.set_trace()
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the
# following shorthand
outdir = os.path.join('output', os.path.splitext(os.path.basename(config_filepath))[0])
filelists = {s: os.path.join(outdir, 'filelist_{}.txt'.format(s))
             for s in ('train', 'val')}

list_of_identifiers = {}
list_of_image_paths = {}
for s in ('val',):
    with open(filelists[s], 'r') as f:
        list_of_image_paths[s] = [s.strip() for s in f.readlines()]
        list_of_identifiers[s] = [os.path.splitext(os.path.basename(fn))[0] for fn in list_of_image_paths[s]]
    for fn in list_of_identifiers[s]:
        assert os.path.exists(outfile_from_filename(fn)), '{} does not exist'.format(outfile_from_filename(fn))


import torch
import cv2
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

exporter = FigExporter()

for s in ('val',):
    random_index = 0
    im_file = list_of_image_paths[s][random_index]
    outfile = outfile_from_filename(im_file)
    res = torch.load(outfile)
    im_id = os.path.splitext(os.path.basename(im_file))[0]
    assert os.path.exists(im_file), '{} does not exist'.format(im_file)
    im = cv2.imread(im_file)

    print(im.shape)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    v = v.draw_instance_predictions(res["instances"].to("cpu"))
    cv2_imshow(v.get_image()[:, :, ::-1])
    exporter.export_gcf('prediction_{}'.format(im_id))
