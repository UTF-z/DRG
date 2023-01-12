import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from yacs.config import CfgNode as CN
from lib.config import get_config


class HPF(nn.Module):

    def __init__(self):
        super(HPF, self).__init__()

    def forward(self, img):

        f = torch.fft.fft2(img)
        fshift = torch.fft.fftshift(f)

        b, c, rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        fshift[:, :, crow - 2:crow + 2, ccol - 2:ccol + 2] = 0

        ishift = torch.fft.ifftshift(fshift)
        iimg = torch.fft.ifft2(ishift)
        iimg = torch.abs(iimg)
        return iimg


class BHF(nn.Module):

    def __init__(self, d=20, n=1, image_shape=[1, 1024, 1024]):
        super(BHF, self).__init__()

        def make_transform_matrix(d):
            transform_matrix = torch.zeros(image_shape)
            center_point = tuple(map(lambda x: (x - 1) / 2, image_shape[-2:]))
            for i in range(transform_matrix.shape[-2]):
                for j in range(transform_matrix.shape[-1]):

                    def cal_distance(pa, pb):
                        import math
                        dis = math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
                        return dis

                    dis = cal_distance(center_point, (i, j))
                    transform_matrix[:, i, j] = 1 / (1 + (d / dis)**(2 * n))
            return transform_matrix

        self.d_matrix = make_transform_matrix(d)
        print(self.d_matrix.shape)

    def forward(self, image):

        f = torch.fft.fft2(image)
        fshift = torch.fft.fftshift(f)

        new_img = torch.abs(
            torch.fft.ifft2(torch.fft.ifftshift(fshift *
                                                self.d_matrix.repeat(image.shape[0], 1, 1, 1).to(image.device))))
        return new_img


class Fswitch(nn.Module):

    def __init__(self):
        super(Fswitch, self).__init__()

    def forward(self, img):
        dft = torch.fft.fft2(img)
        dft_shift = torch.fft.fftshift(dft)
        res1 = 20 * torch.log(dft_shift.norm(dim=1, keepdim=True))
        return res1


class LPF(nn.Module):

    def __init__(self) -> None:
        super(LPF, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (3, 3), padding=1)
        w1 = torch.tensor(np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8],
                                    [1 / 16, 1 / 8, 1 / 16]])).reshape(1, 1, 3, 3).to(torch.float32)
        self.conv1.weight = torch.nn.Parameter(w1)

    def forward(self, img):
        img_blurred = self.conv1(img)
        return img_blurred


class EQH:

    def __init__(self) -> None:
        pass

    def forward(self, img):
        img_aug = cv2.equalizeHist(img)
        return img_aug

    def read(self, path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def write(self, path, img):
        return cv2.imwrite(path, img)


class _Preprocessor(nn.Module):

    def __init__(self, cfg):
        super(_Preprocessor, self).__init__()
        preprocess_types = cfg.PREPROCESS.TYPES
        self.sub_preprocessor = nn.ModuleList()
        if "lpf" in preprocess_types:
            self.sub_preprocessor.add_module('lpf', LPF())
        if 'bhf' in preprocess_types:
            image_shape = cfg.PREPROCESS.IMAGE_SHAPE
            self.sub_preprocessor.add_module('bhf', BHF(d=20, n=1, image_shape=image_shape))
        if "hpf" in preprocess_types:
            self.sub_preprocessor.add_module('hpf', HPF())
        if 'fft' in preprocess_types:
            self.sub_preprocessor.add_module('fft', Fswitch())

    def forward(self, img):
        preprocessed_imgs = [img.to(torch.float32) / 255]
        for preprocess in self.sub_preprocessor:
            preprocessed_imgs.append(preprocess.forward(img).to(torch.float32) / 255)
        result = torch.cat(preprocessed_imgs, dim=1)
        return result


class DummyPreprocessor(nn.Module):

    def __init__(self, cfg):
        super(DummyPreprocessor, self).__init__()

    def forward(self, img):
        return img

class NormalizePreprocessor(nn.Module):

    def __init__(self, cfg, mean, std):
        super(NormalizePreprocessor, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, img):
        return (img - self.mean) / self.std

if __name__ == "__main__":
    cfg = get_config('config/drg_baseline.yml')
    img = cv2.imread("assets/images/train/001.png", cv2.IMREAD_GRAYSCALE)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(torch.float64)
    pre = DummyPreprocessor(cfg)
    out = pre.forward(img)
    for i in range(out.shape[1]):
        cv2.imshow('test', out[0, i, :, :].detach().numpy())
        cv2.waitKey()