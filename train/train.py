import sys
sys.path.append('.')
from lib.dataset.DRGrading import DRGrading
from lib.models.resnetModel import ResnetModel
from lib.metrics.kappa import quadratic_weighted_kappa
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2, os
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import argparse
from yacs.config import CfgNode as CN
from lib.config import get_config
from lib.utils.logger import DRGLogger
from torch.nn.utils import clip_grad
from lib.preprocess.preprocess import DummyPreprocessor
from lib.preprocess.preprocess import NormalizePreprocessor
from lib.const import Queries
from lib.dataset.DRGrading import TripletDataset, TripletDataset_online_augment


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
    epochs = cfg.TRAIN.EPOCHS
    batch_size = cfg.TRAIN.BATCH_SIZE
    normalize = cfg.TRAIN.NORMALIZE
    DRGLogger.warning(f"epochs: {epochs}, batch_size: {batch_size}")
    model_type = 'Attention' if cfg.MODEL.ATTENTION else ''
    preprocess_type = 'Normalize' if normalize else ''
    exp_path = cfg.DATASET.AUG + '-' + model_type + '-' + cfg.LOSS.TYPE + '-' + preprocess_type + '-' + f"{time.strftime('%m-%d-%H-%M-%S', time.localtime())}"
    exp_dir = os.path.join('exp', exp_path)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    cfg_path = os.path.join(exp_dir, "dump_cfg.yml")
    with open(cfg_path, 'w') as f:
        f.write(cfg.dump(sort_keys=False))

    # trip data
    train_data_trip = TripletDataset_online_augment('assets/DRG_data/Triplet', cfg.DATASET.TRAIN_SPLIT, cfg.DATASET.AUG)
    train_data_trip.set_device(device)
    train_trip_loader = DataLoader(
        dataset=train_data_trip,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    val_data_trip = TripletDataset('assets/DRG_data/Triplet', cfg.DATASET.VAL_SPLIT)
    val_data_trip.set_device(device)
    val_trip_loader = DataLoader(
        dataset=val_data_trip,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    mean, std = train_data_trip.get_mean_and_std()
    if normalize:
        DRGLogger.info("Train: Using Normalization")
        preprocessor = NormalizePreprocessor(cfg, mean, std).to(device)
    else:
        preprocessor = DummyPreprocessor(cfg).to(device)
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
    DRGLogger.info(f"the model has {sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)} M params")
    model.to(device)
    kappa_log_path = None
    for epoch in range(start_epoch, epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_trip_loader)):
            batch[Queries.IMG] = preprocessor(batch[Queries.IMG])
            step_idx = step + epoch * len(train_trip_loader)
            optimizer.zero_grad()
            res_dict = model(batch, step_idx, 'train')
            summary.add_scalar(f"resnet_loss", res_dict[Queries.LOSS].item(), step_idx)
            summary.add_scalar(f"acc", res_dict[Queries.ACC].item(), step_idx)
            # summary.add_scalar(f"kappa", res_dict[Queries.KAPPA], step_idx)
            res_dict[Queries.LOSS].backward()
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
            model.eval()
            acc = 0
            total_pred = []
            total_gt = []
            with torch.no_grad():
                for step, batch in enumerate(tqdm(val_trip_loader)):
                    batch[Queries.IMG] = preprocessor(batch[Queries.IMG])
                    label = batch[Queries.LABEL]
                    pred = model(batch, step, 'test')
                    pred = torch.argmax(pred, dim=1)
                    total_pred = np.append(total_pred, pred.cpu().numpy(), axis=0)
                    total_gt = np.append(total_gt, label.cpu().numpy(), axis=0)
            acc = (total_gt == total_pred).sum() / len(total_gt)
            kappa = quadratic_weighted_kappa(total_gt, total_pred)
            kappa_log_path = os.path.join(exp_dir, f"kappa_log.txt")
            with open(kappa_log_path, 'a') as f:
                f.write(f"epoch: {epoch}, kappa = {kappa}\n")
        summary.close()
            
    model.eval()
    acc = 0
    total_pred = []
    total_gt = []
    for step, batch in enumerate(tqdm(val_trip_loader)):
        batch[Queries.IMG] = preprocessor(batch[Queries.IMG])
        label = batch[Queries.LABEL]
        pred = model(batch, step, 'test')
        pred = torch.argmax(pred, dim=1)
        total_pred = np.append(total_pred, pred.cpu().numpy(), axis=0)
        total_gt = np.append(total_gt, label.cpu().numpy(), axis=0)
    print(total_pred, total_gt)
    acc = (total_gt == total_pred).sum() / len(total_gt)
    kappa = quadratic_weighted_kappa(total_gt, total_pred)
    DRGLogger.info(f"Training finished. Test accuracy = {acc}, kappa = {kappa}")
    with open(kappa_log_path, 'a') as f:
        f.write(f"Training finished. Test accuracy = {acc}, kappa = {kappa}")

def exp(config_dir, args):
    config_names = os.listdir(config_dir)
    configs = []
    for name in config_names:
        path = os.path.join(config_dir, name)
        cfg = get_config(path, args)
        configs.append(cfg)
    for cfg in configs:
        main(args, cfg)

def test(args, cfg):
    device = 'cuda:0'
    batch_size = cfg.TRAIN.BATCH_SIZE
    normalize = cfg.TRAIN.NORMALIZE
    model_type = 'Attention' if cfg.MODEL.ATTENTION else ''
    preprocess_type = 'Normalize' if normalize else ''
    test_path = cfg.DATASET.AUG + '-' + model_type + '-' + cfg.LOSS.TYPE + '-' + preprocess_type + '-' + f"{time.strftime('%m-%d-%H-%M-%S', time.localtime())}"
    test_dir = os.path.join('test', test_path)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    cfg_path = os.path.join(test_dir, "dump_cfg.yml")
    with open(cfg_path, 'w') as f:
        f.write(cfg.dump(sort_keys=False))

    # trip data
    mean, std = 0.412984, 0.277988 # magic number
    test_data_trip = TripletDataset('assets/DRG_data/Triplet', cfg.DATASET.TEST_SPLIT)
    test_data_trip.set_device(device)
    test_trip_loader = DataLoader(
        dataset=test_data_trip,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    if normalize:
        DRGLogger.info("Train: Using Normalization")
        preprocessor = NormalizePreprocessor(cfg, mean, std).to(device)
    else:
        preprocessor = DummyPreprocessor(cfg).to(device)
    model = ResnetModel(cfg)
    checkpoint = torch.load(args.test)
    model.load_state_dict(checkpoint['model'])
    DRGLogger.info(f"the model has {sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)} M params")
    model.to(device)
    model.eval()
    total_pred_prob = []
    total_pred_class = []
    total_name = []
    for step, batch in enumerate(tqdm(test_trip_loader)):
        batch[Queries.IMG] = preprocessor(batch[Queries.IMG])
        label = batch[Queries.LABEL]
        name = batch[Queries.NAME]
        pred_prob = model(batch, step, 'test')
        pred_class = torch.argmax(pred_prob, dim=1)
        pred_prob = F.softmax(pred_prob, dim=1)
        total_pred_prob.append(pred_prob.cpu().numpy())
        total_pred_class.append(pred_class.cpu().numpy())
        total_name.append(name)
    total_pred_prob = np.concatenate(total_pred_prob, axis=0)
    total_pred_class = np.concatenate(total_pred_class, axis=0)
    total_name = np.concatenate(total_name, axis=0)
    print(total_name[:10], total_pred_class[:10])
    P0 = total_pred_prob[:, 0]
    P1 = total_pred_prob[:, 1]
    P2 = total_pred_prob[:, 2]
    print(len(total_name), len(total_pred_class), len(P0), len(P1), len(P2))
    test_result = {
        'case': total_name,
        'class': total_pred_class,
        'P0': P0,
        'P1': P1,
        'P2': P2
    }
    test_name = os.path.join(test_dir, 'result.csv')
    dataframe = pd.DataFrame(test_result)
    dataframe.to_csv(test_name, index=False, sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRG trainer')
    parser.add_argument("-b", "--batch_size", type=int, default=None, help='Batch size, override cfg.TRAIN.BATCHSIZE, default None.')
    parser.add_argument("-c", "--config", type=str, default=None, help="Path of a config file.")
    parser.add_argument("--resume", type=str, default=None, help='Path of a specific checkpoint')
    parser.add_argument("--reload", type=str, default=None, help='Not used.')
    parser.add_argument("--test", type=str, default=None, help="The checkpoint that you want to submit")
    parser.add_argument("--exp", type=str, default=None, help="Config directory for batch processing")
    args = parser.parse_args()
    if args.exp is not None:
        exp('config/exp', args)
    else:
        if args.config is None:
            DRGLogger.warning("missing config path, abort.")
            exit()
        cfg = get_config(args.config, args)
        if args.test is not None:
            test(args, cfg)
        else:
            main(args, cfg)
