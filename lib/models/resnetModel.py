from yacs.config import CfgNode as CN
from lib.models.resnet import ResNet, BasicBlock, BottleNeck
import torch.nn as nn
from lib.const import Queries
import torch.nn.functional as F
import torch


class ResnetModel(nn.Module):

    def __init__(self, cfg):
        super(ResnetModel, self).__init__()
        self.num_residuals = cfg.MODEL.NUM_RESIDUALS
        self.classes = cfg.MODEL.CLASSES
        self.resnet = ResNet(BasicBlock, self.num_residuals, self.classes, include_top=True)

    def setup(self, summary):
        self.summary = summary

    def forward(self, batch, step_idx, mode):
        if mode == "train":
            return self.train_step(batch, step_idx)
        if mode == "val":
            return self.val_step(batch, step_idx)
        if mode == "test":
            return self.test_step(batch, step_idx)
        return ValueError(f"Unknow mode {mode}")

    def train_step(self, batch, step_idx):
        imgs = batch[Queries.IMG]
        labels = batch[Queries.LABEL]
        resnet_res = self.resnet(imgs)
        resnet_loss = self.compute_loss(resnet_res, labels)
        with torch.no_grad():
            acc = self.compute_acc(resnet_res, labels)
            acc /= imgs.shape[0]
        self.summary.add_scalar(f"resnet_loss", resnet_loss.item(), step_idx)
        self.summary.add_scalar(f"acc", acc.item(), step_idx)
        return resnet_res, resnet_loss

    def val_step(self, batch, step_idx):
        imgs = batch[Queries.IMG]
        labels = batch[Queries.LABEL]
        resnet_res = self.resnet(imgs)
        resnet_loss = self.compute_loss(resnet_res, labels)
        acc = self.compute_acc(resnet_res, labels)
        return resnet_res, resnet_loss, acc

    def test_step(self, batch, step_idx):
        imgs = batch[Queries.IMG]
        labels = batch[Queries.LABEL]
        resnet_res = self.resnet(imgs)
        return resnet_res

    def compute_loss(self, res, labels):
        return F.cross_entropy(res, labels)

    def compute_acc(self, resnet_res, labels):
        _, class_res = torch.max(resnet_res, dim=1)
        acc = (class_res == labels).sum()
        return acc
