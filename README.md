# DRG

## 1 准备数据集
从官网上下载DR分级的数据集，放入assets文件夹，文件夹结构为
- assets
    - gts
    - images

## 2 准备config
创建config文件夹，里面新建.yml文档，设置config字段。一个示例如下：
```
DATASET:
  TRAIN_SPLIT: train
  VAL_SPLIT: val
  TEST_SPLIT: test
  AUG: raw
  IN_CHANNEL: 1
  IMAGE_SHAPE: [1, 1024, 1024]

PREPROCESS:
  TYPES: [] #'eqh' or 'fft' or 'bhf' or 'hpf' or 'lpf'
  USE_EQH: true

TRAIN:
  SAVE_FREQ: 10
  NORMALIZE: true
  EPOCHS: 200
  BATCH_SIZE: 256
  LEARNING_RATE: 1e-3
  WEIGHT_DECAY: 0.5
  MILESTONE: [10, 80, 100, 150]
  LR_DECAY: 0.1
  GRAD_CLIP:
    NORM: 1.0
    TYPE: 2

MODEL:
  NUM_RESIDUALS: [3, 4, 6, 3]
  CLASSES: 3
  ATTENTION: true

LOSS:
  TYPE: FOCAL # CE or FOCAL
  ALPHA: [0.19697442, 0.30475288, 0.93184052]
```
## 3 离线生成数据
使用命令
```
python lib/dataset/data_processing.py
```
生成数据集
## 4 启动主程序
训练代码：
```
python train/train.py -c /path/to/config
```
测试代码：
```
python train/train.py -c /path/to/config --test /path/to/checkpoint
```