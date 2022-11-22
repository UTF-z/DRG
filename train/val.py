from lib.dataset.DRGrading import DRGrading
from lib.models.resnetModel import ResnetModel
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2, os
from tqdm import tqdm
import time
import argparse
from yacs.config import CfgNode as CN
from lib.config import get_config
from lib.utils.logger import logger
from torch.nn.utils import clip_grad


def main(args, cfg):
    drg_data = DRGrading('assets/images', 'assets/gts', cfg.DATASET.SPLIT, cfg.DATASET.MODE, cfg.DATASET.AMOUNT)
    drg_data.set_device('cuda:0')
    epochs = cfg.TRAIN.EPOCHS
    batch_size = cfg.TRAIN.BATCH_SIZE

    dataloader = DataLoader(
        dataset=drg_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    model = ResnetModel(cfg)
    if args.load is not None:
        path = os.path.join(args.load, "state_dict.pt")
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
    logger.info(f"the model has {sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)} M params")
    model.eval()
    model.to('cuda:0')
    total_loss = 0.0
    total_acc = 0
    steps = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            steps += 1
            res, loss, acc = model(batch, steps, 'val')
            total_loss += loss
            total_acc += acc

    avg_loss = total_loss / steps
    avg_acc = total_acc / len(drg_data)
    logger.warning(f"average acc is {avg_acc}")
    logger.warning(f"average loss is {avg_loss}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRG trainer')
    parser.add_argument("--reload", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()
    cfg = get_config(args.cfg, args)

    main(args, cfg)