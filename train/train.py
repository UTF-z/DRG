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
    drg_data = DRGrading('assets/images', 'assets/gts', cfg.DATASET.SPLIT, cfg.DATASET.MODE, cfg.DATASET.AMOUNT)
    drg_data.set_device('cuda:0')
    epochs = cfg.TRAIN.EPOCHS
    batch_size = cfg.TRAIN.BATCH_SIZE
    logger.warning(f"epochs: {epochs}, batch_size: {batch_size}")

    dataloader = DataLoader(
        dataset=drg_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = ResnetModel(cfg)
    if args.resume is not None:
        path = os.path.join(args.resume, "state_dict.pt")
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=cfg.TRAIN.MILESTONE,
                                                     gamma=cfg.TRAIN.LR_DECAY)
    exp_path = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
    exp_dir = os.path.join('exp', exp_path)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    summary = SummaryWriter(os.path.join(exp_dir, 'runs'))
    model.setup(summary)
    logger.info(f"the model has {sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)} M params")
    model.train()
    model.to('cuda:0')
    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(dataloader)):
            step_idx = step + epoch * len(dataloader)
            optimizer.zero_grad()
            res, loss = model(batch, step_idx, 'train')
            loss.backward()
            if cfg.TRAIN.GRAD_CLIP_ENABLED:
                clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)
            optimizer.step()
        scheduler.step()

    summary.close()
    model_path = os.path.join(exp_dir, 'state_dict.pt')
    cfg_path = os.path.join(exp_dir, "dump_cfg.yml")
    with open(cfg_path, 'w') as f:
        f.write(cfg.dump(sort_keys=False))
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRG trainer')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reload", type=str, default=None)
    args = parser.parse_args()
    cfg = get_config(args.cfg, args)

    main(args, cfg)