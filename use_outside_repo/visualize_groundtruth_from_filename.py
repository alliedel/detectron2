import numpy as np
from PIL import Image
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import Visualizer
import os
import tempfile

from export_proposal_helpers import get_maskrcnn_cfg, cv2_imshow, FigExporter


def get_splits_and_ids_for_image_ids(image_ids, list_of_image_ids):
    corresponding_splits = []
    indices_into_dataset = []
    for image_id in image_ids:
        s = None
        index_into_dataset = None
        for split, imagelist in list_of_image_ids.items():
            if image_id in imagelist:
                print('{} in {}'.format(image_id, split))
                s = split
                index_into_dataset = imagelist.index(image_id)
                print('Index into dataset:', index_into_dataset)
        if s is None:
            import ipdb;
            ipdb.set_trace()
            raise Exception('{} not found in list of images')
        indices_into_dataset.append(index_into_dataset)
        corresponding_splits.append(s)
    return corresponding_splits, indices_into_dataset


def main():
    cfg = get_maskrcnn_cfg()
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    scale = 2.0

    exporter = FigExporter()
    i = 0
    done = False

    image_ids = ('9', '306284', '389490')
    n_images_to_export = len(image_ids)

    # dataloaders = {
    #     s: DefaultTrainer.build_test_loader(cfg, {'train': cfg.DATASETS.TRAIN[0], 'val': cfg.DATASETS.TEST[0]}[s]) for s
    #     in ('train', 'val')
    # }
    dataloaders = {
        'train': build_detection_train_loader(cfg),
        'val': []
    }

    try:
        list_of_image_ids = {}
        for split in ('train', 'val'):
            with open('output/mask_rcnn_R_50_FPN_3x/filelist_{}.txt'.format(split), 'r') as f:
                list_of_image_ids[split] = [str(os.path.splitext(os.path.basename(s.strip()))[0]).lstrip('0') for s in
                                            f.readlines()]
        corresponding_splits, indices_into_dataset = get_splits_and_ids_for_image_ids(image_ids, list_of_image_ids)
        for s, index_into_dataset, image_id in zip(corresponding_splits, indices_into_dataset, image_ids):
            per_image = dataloaders[s].dataset[index_into_dataset]
            assert per_image['image_id'] == image_id
    except AssertionError:
        print('Generating list of image ids')
        list_of_image_ids = {'train': [],
                             'val': []
                             }
        for i, d in enumerate(dataloaders['train'].dataset):
            if i % 10 == 0:
                print(i, '/', len(dataloaders['train'].dataset))
            list_of_image_ids['train'].append(d['image_id'])
        fname = tempfile.NamedTemporaryFile().name
        with open(fname, 'w') as f:
            f.write('\n'.join(list_of_image_ids['train']) + '\n')
        print('Done generating list of image ids.  Saved to {}.'.format(fname))
        corresponding_splits, indices_into_dataset = get_splits_and_ids_for_image_ids(image_ids, list_of_image_ids)

    print('Found corresponding indices into datasets.')

    for s, index_into_dataset, image_id in zip(corresponding_splits, indices_into_dataset, image_ids):
        per_image = dataloaders[s].dataset[index_into_dataset]
        img = per_image["image"].permute(1, 2, 0)
        if cfg.INPUT.FORMAT == "BGR":
            img = img[:, :, [2, 1, 0]]
        else:
            img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))

        visualizer = Visualizer(img, metadata=metadata, scale=scale)
        visualizer._default_font_size = visualizer._default_font_size * 1.5
        target_fields = per_image["instances"].get_fields()
        labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
        vis = visualizer.overlay_instances(
            labels=labels,
            boxes=target_fields.get("gt_boxes", None),
            masks=None,
            keypoints=None,
        )
        cv2_imshow(vis.get_image()[:, :, ::-1])
        exporter.export_gcf(tag='gt_' + str(per_image["image_id"]))


if __name__ == '__main__':
    main()
