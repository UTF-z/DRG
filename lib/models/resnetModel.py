from yacs.config import CfgNode as CN
from lib.models.resnet import ResNet, BasicBlock, AttentionBlock
import torch.nn as nn
from lib.const import Queries
import torch.nn.functional as F
import torch
from lib.metrics.kappa import quadratic_weighted_kappa
from lib.utils.logger import DRGLogger

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
        self.attention = cfg.MODEL.ATTENTION
        if cfg.LOSS.TYPE == 'CE':
            self.loss_function = self.cross_entropy_loss
            DRGLogger.info('LOSS: Using Cross Entropy Loss')
        elif cfg.LOSS.TYPE == 'FOCAL':
            self.loss_function = self.focal_loss
            self.alpha = torch.tensor(cfg.LOSS.ALPHA)
            DRGLogger.info('LOSS: Using Focal Loss')
        if not self.attention:
            DRGLogger.info('ATTENTION: No Self Attention')
            self.resnet = ResNet(BasicBlock,
                                self.num_residuals,
                                self.classes,
                                include_top=True,
                                input_channel=cfg.DATASET.IN_CHANNEL)
        else:
            DRGLogger.info('ATTENTION: Using Self Attention')
            self.resnet = ResNet(AttentionBlock,
                                 self.num_residuals,
                                 self.classes,
                                 include_top=True,
                                 input_channel=cfg.DATASET.IN_CHANNEL)

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
        # kappa = self.compute_kappa(resnet_res, labels)
        res_dict = {
            Queries.RES: resnet_res,
            Queries.LOSS: resnet_loss,
            Queries.ACC: acc,
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
        resnet_res = self.resnet(imgs)
        return resnet_res
    
    def compute_loss(self, res, labels):
        loss = self.loss_function(res, labels)
        return loss
    
    def cross_entropy_loss(self, pred, gt):
        loss = F.cross_entropy(pred, gt)
        return loss
    
    def focal_loss(self, pred, gt):
        mask = torch.ones_like(pred)
        flabels = torch.zeros_like(pred)
        flabels = torch.scatter(flabels, 1, gt.unsqueeze(1), 1)
        loss = multiclass_focal_loss(pred, flabels, mask, alpha=self.alpha, reduction='mean')
        return loss
    
    @torch.no_grad()
    def compute_acc(self, resnet_res, labels) -> torch.Tensor:
        class_res = torch.argmax(resnet_res, dim=1)
        acc = (class_res == labels).sum()
        acc = acc.true_divide(len(labels))
        return acc

    @torch.no_grad()
    def compute_kappa(self, resnet_res, labels) -> torch.Tensor:
        class_res = (torch.argmax(resnet_res, dim=1)).cpu().numpy()
        labels = labels.cpu().numpy()
        kappa = quadratic_weighted_kappa(labels, class_res)
        return kappa