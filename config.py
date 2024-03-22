import os

TRAINSETS                   = ["Cityscapes+COCO"]
VALSETS                     = ["LostAndFound", "Fishyscapes"]
MODELS                      = ["DeepLabV3+_WideResNet38", "DualGCNNet_res50"]

TRAINSET                    = TRAINSETS[0]
VALSET                      = VALSETS[0]
MODEL                       = MODELS[0]

ROOT                        = os.path.dirname(os.path.realpath(__file__))
DATASETS_ROOT               = os.path.join(ROOT, "datasets")
IO_ROOT                     = os.path.join(ROOT, "io")
PRETRAINED_WEIGHTS_ROOT     = os.path.join(ROOT, "weights")
CHECKPOINTS_ROOT            = os.path.join(IO_ROOT, "checkpoints")
CS_ROOT                     = os.path.join(DATASETS_ROOT, "cityscapes")
COCO_ROOT                   = os.path.join(DATASETS_ROOT, "COCO", "2017")
LAF_ROOT                    = os.path.join(DATASETS_ROOT, "lost_and_found")
FS_ROOT                     = os.path.join(DATASETS_ROOT, "fishyscapes")

class cs_coco_roots:
    """
    OoD training roots for Cityscapes + COCO mix
    """
    model_name  = MODEL
    init_ckpt   = os.path.join(PRETRAINED_WEIGHTS_ROOT, model_name, ".pth")
    cs_root     = CS_ROOT
    coco_root   = COCO_ROOT
    io_root     = os.path.join(IO_ROOT, model_name)
    weights_dir = CHECKPOINTS_ROOT

class laf_roots:
    """
    LostAndFound config class
    """
    model_name = MODEL
    init_ckpt = os.path.join(PRETRAINED_WEIGHTS_ROOT, model_name, ".pth")
    eval_dataset_root = LAF_ROOT
    io_root = os.path.join(IO_ROOT, model_name, "laf_eval")
    weights_dir = CHECKPOINTS_ROOT


class fs_roots:
    """
    Fishyscapes config class
    """
    model_name = MODEL
    init_ckpt = os.path.join(PRETRAINED_WEIGHTS_ROOT, model_name, ".pth")
    eval_dataset_root = FS_ROOT
    io_root = os.path.join(IO_ROOT, model_name, "fs_eval")
    weights_dir = CHECKPOINTS_ROOT


class params:
    """
    Set pipeline parameters
    """
    training_starting_epoch = 0
    num_training_epochs     = 1
    pareto_alpha            = 0.9
    ood_subsampling_factor  = 0.1
    learning_rate           = 1e-5
    crop_size               = 480
    val_epoch               = num_training_epochs
    batch_size              = 8
    entropy_threshold       = 0.7
