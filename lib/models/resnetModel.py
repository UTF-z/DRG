from yacs.config import CfgNode as CN
from lib.models.resnet import ResNet, BasicBlock, BottleNeck
import torch.nn as nn
from lib.const import Queries
import torch.nn.functional as F
import torch

def multiclass_focal_loss(
    inputs: torch.Tensor,  # TENSOR (B x N, C)
    targets: torch.Tensor,  # TENSOR (B x N, C)
    masks: torch.Tensor,  # TENSOR (B x N, 1)
    alpha=None,  # TENSOR (C, 1)
    gamma: float = 2,
    reduction: str = "none",
):
    if masks.sum().detach().cpu().item() != 0:
        n_classes = inputs.shape[1]
        logit = F.softmax(inputs, dim=1)  # TENSOR (B x N, C)
        if alpha is None:
            alpha = torch.ones((n_classes), requires_grad=False)

        if alpha.device != inputs.device:
            alpha = alpha.to(inputs.device)

        epsilon = 1e-10
        pt = torch.sum((targets * logit), dim=1, keepdim=True) + epsilon  # TENSOR (B x N, 1)
        log_pt = pt.log()  # TENSOR (B x N, 1)

        targets_idx = torch.argmax(targets, dim=1, keepdim=True).long()  # TENSOR (B x N, 1)
        alpha = alpha[targets_idx]  # TENSOR ( B x N, 1)

        focal_loss = -1 * alpha * (torch.pow((1 - pt), gamma) * log_pt)  # TENSOR (B x N, 1)
        masked_focal_loss = focal_loss * masks  # TENSOR (B x N, 1)

        if reduction == "mean":
            loss = masked_focal_loss.sum() / masks.sum()
        elif reduction == "sum":
            loss = masked_focal_loss.sum()
        else:
            loss = masked_focal_loss

        return loss
    else:
        return torch.Tensor([0.0]).float().to(inputs.device)

class ResnetModel(nn.Module):

    def __init__(self, cfg):
        super(ResnetModel, self).__init__()
        self.num_residuals = cfg.MODEL.NUM_RESIDUALS
        self.classes = cfg.MODEL.CLASSES
        self.preprocessor = None
        self.alpha = torch.tensor(cfg.MODEL.ALPHA)
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
        mask = torch.ones_like(res)
        flabels = torch.zeros_like(res)
        flabels = torch.scatter(flabels, 1, labels.unsqueeze(1), 1)
        loss = multiclass_focal_loss(res, flabels, mask, alpha=self.alpha, reduction='sum')
        # loss = F.cross_entropy(res, labels)
        return loss

    @torch.no_grad()
    def compute_acc(self, resnet_res, labels) -> torch.Tensor:
        class_res = torch.argmax(resnet_res, dim=1)
        acc = (class_res == labels).sum()
        acc = acc.true_divide(len(labels))
        return acc
