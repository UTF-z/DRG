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
        self.preprocessor = None
        self.resnet = ResNet(BasicBlock,
                             self.num_residuals,
                             self.classes,
                             include_top=True,
                             input_channel=len(cfg.PREPROCESS.TYPES) + 1)

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
        acc = self.compute_acc(resnet_res, labels)
        res_dict = {
            Queries.RES: resnet_res,
            Queries.LOSS: resnet_loss,
            Queries.ACC: acc
        }
        return res_dict

    @torch.no_grad()
    def val_step(self, batch, step_idx):
        imgs = batch[Queries.IMG]
        labels = batch[Queries.LABEL]
        resnet_res = self.resnet(imgs)
        resnet_loss = self.compute_loss(resnet_res, labels)
        acc = self.compute_acc(resnet_res, labels)
        res_dict = {
            Queries.RES: resnet_res,
            Queries.LOSS: resnet_loss,
            Queries.ACC: acc
        }
        return res_dict

    @torch.no_grad()
    def test_step(self, batch, step_idx):
        imgs = batch[Queries.IMG]
        labels = batch[Queries.LABEL]
        resnet_res = self.resnet(imgs)
        return resnet_res

    def compute_loss(self, res, labels):
        return F.cross_entropy(res, labels)

    @torch.no_grad()
    def compute_acc(self, resnet_res, labels) -> torch.Tensor:
        class_res = torch.argmax(resnet_res, dim=1)
        acc = (class_res == labels).sum()
        acc = acc.true_divide(len(labels))
        return acc
