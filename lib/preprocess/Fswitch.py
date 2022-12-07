import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

class Fswitch:
    
    def FFT(self,img):
        rows, cols = img.shape[:2]
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        res1 = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return res1


    def read(self,path):
        return cv2.imread(path,cv2.IMREAD_GRAYSCALE)


    def write(self,path,img):
        return cv2.imwrite(path,img)

    def __init__(self, img_dir, gt_dir, split):
        self.img = []
        fft_dir = os.path.join(img_dir,'processed','Fswitch')
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
            self.write(os.path.normpath(os.path.join(fft_dir,image_name)),self.FFT(img))

if __name__ == "__main__":
    img_path = os.path.join('assets', 'images')
    gts_path = os.path.join('assets', 'gts')
    fft = Fswitch(img_path,gts_path,'train')
