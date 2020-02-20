from detectron2.config import get_cfg
import subprocess
import os
import torch, torch.distributed, logging, time
from detectron2.evaluation.evaluator import inference_context
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt

DETECTRON_MODEL_ZOO = os.path.expanduser('~/data/models/detectron_model_zoo')
assert os.path.isdir(DETECTRON_MODEL_ZOO)
DETECTRON_REPO = './detectron2_repo'


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def download_detectron_model_to_local_zoo(relpath):
    if relpath.startswith('detectron2://'):
        relpath.replace('detectron2://', '', 1)
    url = 'https://dl.fbaipublicfiles.com/detectron2/' + relpath
    outpath = os.path.join(DETECTRON_MODEL_ZOO, relpath)
    outdir = os.path.dirname(outpath)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    assert os.path.isdir(outdir)
    if os.path.exists(outpath):
        return outpath
    try:
        stdout, stderr = subprocess.check_call(['wget', url, '-O', outpath])
    except subprocess.CalledProcessError:
        # print(stderr)
        raise
    return outpath


def get_maskrcnn_cfg(config_filepath=f"{DETECTRON_REPO}/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
    cfg = get_cfg()
    cfg.merge_from_file(config_filepath)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the
    # following shorthand
    model_rel_path = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
    local_path = download_detectron_model_to_local_zoo(model_rel_path)
    cfg.MODEL.WEIGHTS = local_path
    return cfg


def just_inference_on_dataset(model, data_loader, outdir, stop_after_n_points=None, get_proposals=False):
    """
    Function by Allie.

    Run model (in eval mode) on the data_loader and evaluate the metrics with evaluator.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    n_points = len(data_loader)
    proposals_outdir = os.path.join(outdir, 'proposals') if get_proposals else None
    if get_proposals:
        if os.path.exists(proposals_outdir):
            raise Exception('Proposals outdir {} already exists.  Please delete.')
        os.makedirs(proposals_outdir)

    inference_outdir = os.path.join(outdir, 'predictions')
    if os.path.exists(inference_outdir):
        raise Exception('Predictions outdir {} already exists.  Please delete.')
    os.makedirs(inference_outdir)
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):

            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0
            elif stop_after_n_points is not None and idx >= stop_after_n_points:
                break

            start_compute_time = time.time()
            outputs = model(inputs)

            if get_proposals:
                images = model.preprocess_image(inputs)
                features = model.backbone(images.tensor)
                proposalss, proposal_lossess = model.proposal_generator(images, features, None)

            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            print(idx, '/', n_points)
            if data_loader.batch_sampler.batch_size != 1:
                raise NotImplementedError
            else:
                assert len(outputs) == 1
                assert len(inputs) == 1
            for output, input in zip(outputs, inputs):
                output.update({
                    k: input[k] for k in ('file_name', 'image_id')
                })
                torch.save(output, os.path.join(inference_outdir, 'output_' +
                                                os.path.splitext(os.path.basename(input['file_name']))[0] + '.pt'))
            if get_proposals:
                for proposals, input in zip(proposalss, inputs):
                    torch.save(proposals, os.path.join(outdir, 'proposals', 'proposals_' +
                                                       os.path.splitext(os.path.basename(input['file_name']))[0] + '.pt'))
    return


def proposal_predictor_forward_pass(predictor, batched_inputs):
    """
    Instead of running the forward pass of the full R-CNN model, we extract the proposals.
    """

    images = predictor.model.preprocess_image(batched_inputs)
    features = predictor.model.backbone(images.tensor)
    proposals, proposal_losses = predictor.model.proposal_generator(images, features, None)
    return proposals


def get_image_identifiers(data_loader, identifier_strings=('file_name', 'image_id'), n_images_stop=None):
    assert data_loader.batch_sampler.batch_size == 1, NotImplementedError('Only handing case of batch size = 1 for now')
    assert isinstance(data_loader.sampler, torch.utils.data.sampler.SequentialSampler), \
        'The data loader is not sequential, so the ordering will not be consistent if I give you the filenames.  ' \
        'Choose a data loader with a sequential sampler.'
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    total = len(data_loader)  # inference data loader must have a fixed length

    logging_interval = 50
    all_identifiers = []
    print('Collecting image identifiers')
    for idx, inputs in enumerate(data_loader):
        if idx % 10 == 0:
            print(idx, '/', n_images_stop or len(data_loader))
        x = inputs[0]
        if n_images_stop is not None and idx >= n_images_stop:
            break
        all_identifiers.append(
            {s: x[s] for s in identifier_strings}
        )

    return all_identifiers


def display_instances_on_image(image, instance_output_dict, cfg):
    """
    :param image: numpy array (HxWx3)
    :param instance_output_dict: {
                                  'pred_boxes': (n_instances,4),
                                  'scores': (n_instances,),
                                  'pred_classes': (n_instances,),
                                  'pred_masks': (n_instances, H, W)
                                  }
    :return: Nothing.  Displays on current cv figure.

    """
    v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(instance_output_dict["instances"].to("cpu"))
    cv2_imshow(v.get_image()[:, :, ::-1])


def cv2_imshow(img):
    h = plt.imshow(img)
    return h


class FigExporter(object):
    fig_number = 1
    workspace_dir = '/home/adelgior/workspace/images'
    generated_figures = []

    def export_gcf(self, tag=None, use_number=True):

        if tag is None:
            assert use_number
            basename = '{:06d}.png'.format(self.fig_number)
        else:
            if use_number:
                basename = '{:06d}_{}.png'.format(self.fig_number, tag)
            else:
                basename = '{}.png'.format(tag)

        fname = os.path.join(self.workspace_dir, basename)

        FigExporter.fig_number += 1
        plt.savefig(fname)
        dbprint('Exported {}'.format(fname))
        self.generated_figures.append(fname)
