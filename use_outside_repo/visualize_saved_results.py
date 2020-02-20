import os
import torch
import cv2
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer


from export_proposal_helpers import cv2_imshow, FigExporter, get_maskrcnn_cfg

with open('output/mask_rcnn_R_50_FPN_3x/filelist_train.txt', 'r') as f:
    image_filenames = [s.strip() for s in f.readlines()]
image_ids = [os.path.splitext(os.path.basename(s))[0].lstrip('0') for s in image_filenames]

# index = 0
# image_id = os.path.splitext(os.path.basename(image_filenames[index]))[0]

# base_image_id = '306284'
base_image_id = '486536'  # 2 cats
# base_image_id = '9'
index = image_ids.index(base_image_id)
image_id = os.path.splitext(os.path.basename(image_filenames[index]))[0]
# image_ids = ('9', '306284', '389490')


prediction_file = 'output/mask_rcnn_R_50_FPN_3x/predictions/output_{}.pt'.format(image_id)
proposals_file = 'output/mask_rcnn_R_50_FPN_3x/proposals/proposals_{}.pt'.format(image_id)

exporter = FigExporter()

predictions = torch.load(prediction_file)
proposals = torch.load(proposals_file)
img = cv2.imread(image_filenames[index])

cfg = get_maskrcnn_cfg()
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
scale = 2.0
v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
v._default_font_size = v._default_font_size * 1.5
v = v.draw_instance_predictions(predictions["instances"].to("cpu"))
cv2_imshow(v.get_image()[:, :, ::-1])
exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_prediction')

cv2_imshow(img[:, :, ::-1])
exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_input')

v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
v._default_font_size = v._default_font_size * 1.5
proposals.pred_boxes = proposals.proposal_boxes
v = v.draw_instance_predictions(proposals.to('cpu'))
cv2_imshow(v.get_image()[:, :, ::-1])
exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_proposals')
