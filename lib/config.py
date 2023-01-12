from argparse import Namespace

from yacs.config import CfgNode as CN

from .utils.logger import DRGLogger

_C = CN(new_allowed=True)

_C.DATA_PRESET = CN(new_allowed=True)

_C.DATASET = CN(new_allowed=True)
_C.DATASET.TRAIN = CN(new_allowed=True)
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.MANUAL_SEED = 1
_C.TRAIN.CONV_REPEATABLE = True
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.EPOCH = 100
_C.TRAIN.OPTIMIZER = "Adam"
_C.TRAIN.LR = 0.001
_C.TRAIN.SCHEDULER = "StepLR"
_C.TRAIN.LR_DECAY_GAMMA = 0.1
_C.TRAIN.LR_DECAY_STEP = [70]
_C.TRAIN.LOG_INTERVAL = 50
_C.TRAIN.FIND_UNUSED_PARAMETERS = False

_C.TRAIN.GRAD_CLIP_ENABLED = True
_C.TRAIN.GRAD_CLIP = CN(new_allowed=True)
_C.TRAIN.GRAD_CLIP.TYPE = 2
_C.TRAIN.GRAD_CLIP.NORM = 0.001

_C.MODEL = CN(new_allowed=True)


def default_config() -> CN:
    """
    Get a yacs CfgNode object with the default config values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def get_config(config_file: str, arg: Namespace = None, merge: bool = True) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
      merge (bool): Whether to merge with the default config or not.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    if merge:
        cfg = default_config()
    else:
        cfg = CN(new_allowed=True)
    cfg.merge_from_file(config_file)

    if arg is not None:
        # if arg.batch_size is given, it always have higher priority
        if arg.batch_size is not None:
            DRGLogger.warning(f"cfg's batch_size {cfg.TRAIN.BATCH_SIZE} reset to arg.batch_size: {arg.batch_size}")
            cfg.TRAIN.BATCH_SIZE = arg.batch_size
        else:
            DRGLogger.info(f"arg.batch_size is None, using cfg's batch_size: {cfg.TRAIN.BATCH_SIZE}")
            arg.batch_size = cfg.TRAIN.BATCH_SIZE

        # if arg.reload is given, it always have higher priority.
        if arg.reload is not None:
            DRGLogger.warning(f"cfg MODEL's pretrained {cfg.MODEL.PRETRAINED} reset to arg.reload: {arg.reload}")
            cfg.MODEL.PRETRAINED = arg.reload

    cfg.freeze()
    return cfg


if __name__ == "__main__":
    cfg: CN = get_config("config/train_bihand2d_fh_pl.yml")
    print(cfg)
    cfg_str = cfg.dump(sort_keys=False)
    with open("tmp/test_dump_cfg.yaml", "w") as f:
        f.write(cfg_str)
