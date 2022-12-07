import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

class HPF:

    def HighPassFilter(self,img):

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        fshift[crow - 2:crow + 2, ccol - 2:ccol + 2] = 0

        ishift = np.fft.ifftshift(fshift)
        iimg = np.fft.ifft2(ishift)
        iimg = np.abs(iimg)
        return iimg


    def read(self,path):
        return cv2.imread(path,cv2.IMREAD_GRAYSCALE)


    def write(self,path,img):
        return cv2.imwrite(path,img)

    def __init__(self, img_dir, gt_dir, split):
        self.img = []
        fft_dir = os.path.join(img_dir,'processed','HPF')
        if split == 'train':
            img_dir = os.path.join(img_dir, 'train')
        elif split == 'test':
            img_dir = os.path.join(img_dir, 'test')
        elif split == 'val':
            img_dir = os.path.join(img_dir, 'val')
        gt_files = os.listdir(gt_dir)
        self.gtfile = pd.read_csv(os.path.join(gt_dir, gt_files[0]))
        for image_name in self.gtfile['image name']:
            img_path = os.path.normpath(os.path.join(img_dir, image_name))
            img = self.read(img_path)
            self.write(os.path.normpath(os.path.join(fft_dir,image_name)),self.HighPassFilter(img))

if __name__ == "__main__":
    img_path = os.path.join('assets', 'images')
    gts_path = os.path.join('assets', 'gts')
    fft = HPF(img_path,gts_path,'train')
