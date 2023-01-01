import sys
sys.path.append('.')
from lib.dataset.DRGrading import DRGrading
from lib.models.resnetModel import ResnetModel
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2, os
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
from yacs.config import CfgNode as CN
from lib.config import get_config
from lib.utils.logger import logger
from torch.nn.utils import clip_grad
from lib.preprocess.preprocess import Preprocessor
from lib.const import Queries


def clip_gradient(optimizer, max_norm, norm_type):
    """Clips gradients computed during backpropagation to avoid explosion of gradients.

    Args:
        optimizer (torch.optim.optimizer): optimizer with the gradients to be clipped
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            clip_grad.clip_grad_norm_(param, max_norm, norm_type)


def main(args, cfg):

    device = 'cuda:0'
    drg_data = DRGrading('assets/images', 'assets/gts', cfg.DATASET.SPLIT, cfg.DATASET.MODE, cfg.DATASET.AMOUNT)
    drg_data.set_device(device)
    epochs = cfg.TRAIN.EPOCHS
    batch_size = cfg.TRAIN.BATCH_SIZE
    logger.warning(f"epochs: {epochs}, batch_size: {batch_size}")
    exp_path = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
    exp_dir = os.path.join('exp', exp_path)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    cfg_path = os.path.join(exp_dir, "dump_cfg.yml")
    with open(cfg_path, 'w') as f:
        f.write(cfg.dump(sort_keys=False))

    dataloader = DataLoader(
        dataset=drg_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    preprocessor = Preprocessor(cfg).to(device)
    model = ResnetModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=cfg.TRAIN.MILESTONE,
                                                     gamma=cfg.TRAIN.LR_DECAY)
    summary = SummaryWriter(os.path.join(exp_dir, 'runs'))
    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
    logger.info(f"the model has {sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)} M params")
    model.train()
    model.to(device)
    for epoch in range(start_epoch, epochs):
        for step, batch in enumerate(tqdm(dataloader)):
            batch[Queries.IMG] = preprocessor(batch[Queries.IMG])
            step_idx = step + epoch * len(dataloader)
            optimizer.zero_grad()
            res, loss, acc = model(batch, step_idx, 'train')
            summary.add_scalar(f"resnet_loss", loss.item(), step_idx)
            summary.add_scalar(f"acc", acc.item(), step_idx)
            loss.backward()
            if cfg.TRAIN.GRAD_CLIP_ENABLED:
                clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)
            optimizer.step()
        scheduler.step()
        if epoch % cfg.TRAIN.SAVE_FREQ == 0:
            checkpoint = {
                'epoch': epoch+1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            checkpoint_path = os.path.join(exp_dir, f'checkpoint_{epoch}.pth.tar')
            torch.save(checkpoint, checkpoint_path)

    summary.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRG trainer')
    parser.add_argument("-b", "--batch_size", type=int, default=None, help='Batch size, override cfg.TRAIN.BATCHSIZE, default None.')
    parser.add_argument("-c", "--config", type=str, default=None, help="Path of a config file.")
    parser.add_argument("--resume", type=str, default=None, help='Path of a specific checkpoint')
    parser.add_argument("--reload", type=str, default=None, help='Not used.')
    args = parser.parse_args()
    if args.config is None:
        logger.warning("missing config path, abort.")
        exit()
    cfg = get_config(args.config, args)

    main(args, cfg)