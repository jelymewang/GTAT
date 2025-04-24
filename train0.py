"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import GOT10k_dataset
from torch.utils.data import DataLoader
from pysot.core.config import cfg
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder
from common_path import *
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
import os
import importlib

# 将pytracking目录添加到sys.path
pytracking_path = "/data_B/renjie/pytracking"
sys.path.append(pytracking_path)
from pytracking.evaluation.tracker import *

# 使用pytracking中的功能


if __name__ == '__main__':

    snapshot_path = os.path.join(project_path_, 'pysot/experiments/%s/model.pth'% siam_model_)
    config_path = os.path.join(project_path_, 'pysot/experiments/%s/config.yaml' % siam_model_)
    # load config
    cfg.merge_from_file(config_path)
    ''' create and initialize model'''
    model = ModelBuilder()

    # 使用pytracking中的功能
    Tracker = Tracker('dimp', 'dimp50')
    params = Tracker.get_parameters()
    dimp = Tracker.create_tracker(params)

    # load model
    model = load_pretrain(model, snapshot_path).cuda().eval()

    ''' create dataset'''
    dataset = GOT10k_dataset(max_num=20)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=8)
    dataset_size = len(dataloader)             # the total number of training iterations

    for epoch in range(1, 2):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        sim_list=[]
        for i, data in enumerate(dataloader):  # inner loop within one epoch
            print(i)
            template = data[0].squeeze(0).cuda()  # pytorch tensor, shape=(1,3,127,127)

            search1 = data[1].squeeze(0).cuda()
            search2 = torch.nn.functional.interpolate(search1, size=(271, 271), mode='bilinear')# pytorch tensor, shape=(N,3,255,255) [0,255]


            def compute_top_k_activated_channel_similarity(feature_map1, feature_map2, k=3, threshold=1e-5):
                """
                计算被激活的最有用的前 k 个通道之间的空间分布相似性。

                Args:
                    feature_map1: 第一个特征图，形状为 (N, C, H, W)。
                    feature_map2: 第二个特征图，形状为 (N, C, H, W)。
                    k: 选取的最有用的通道数量。
                    threshold: 判断通道是否激活的阈值。

                Returns:
                    similarities: 激活通道的相似度矩阵，形状为 (N, k)。
                """
                # 假设 feature_map1 和 feature_map2 的形状都是 (N, C, H, W)
                N, C, H, W = feature_map1.shape

                # 初始化用于存储相似度的张量
                similarities = torch.zeros(N, k)

                for n in range(N):
                    # 计算每个通道的激活强度（这里使用绝对值的均值作为强度）
                    activation_strength1 = torch.mean(torch.abs(feature_map1[n]), dim=(1, 2))  # (C,)
                    activation_strength2 = torch.mean(torch.abs(feature_map2[n]), dim=(1, 2))  # (C,)

                    # 选取被激活的通道（大于阈值）
                    activated_indices1 = (activation_strength1 > threshold).nonzero(as_tuple=True)[0]
                    activated_indices2 = (activation_strength2 > threshold).nonzero(as_tuple=True)[0]

                    # 找到共同激活的通道
                    common_activated_indices = torch.intersect1d(activated_indices1, activated_indices2)

                    # 根据激活强度对这些通道进行排序并选取前 k 个通道
                    if len(common_activated_indices) > 0:
                        # 计算平均激活强度并排序
                        common_activation_strength = (activation_strength1[common_activated_indices] +
                                                      activation_strength2[common_activated_indices]) / 2
                        top_k_indices = common_activated_indices[
                            torch.topk(common_activation_strength, min(k, len(common_activated_indices)))[1]]

                        # 计算相似度
                        for i, c in enumerate(top_k_indices):
                            # 取出第 n 个样本的第 c 个通道，并展平为一维
                            channel1 = feature_map1[n, c].flatten()  # 变为 (H * W,)
                            channel2 = feature_map2[n, c].flatten()  # 变为 (H * W,)

                            # 计算余弦相似度
                            similarity = F.cosine_similarity(channel1, channel2, dim=0)
                            similarities[n, i] = similarity

                return similarities
            model.template(template)
            da_model.(template)
            pp_f=model.backbone(search1)[-1]
            da_f=da_model.featureExtract(search2)
            sim=compute_top_k_activated_channel_similarity(pp_f,da_f)
            sim_list.append(sim)
        average_similarity = np.mean(sim_list)
        print(average_similarity)
