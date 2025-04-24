# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
from os.path import join
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str


from common_path import *
import sys
import importlib


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--tracker_name', default=siam_model_, type=str)
parser.add_argument('--dataset', default=dataset_name_, type=str,
                    help='eval one special dataset')
parser.add_argument('--video', default=video_name_, type=str,
                    help='eval one special video')
parser.add_argument('--vis', default=False, action='store_true',
                    help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)
# 将pytracking目录添加到sys.path
pytracking_path = "/data_B/renjie/pytracking"
sys.path.append(pytracking_path)
from pytracking.evaluation.tracker import *


def compute_top_k_activated_channel_similarity(feature_map1, feature_map2, k=30, target_size=(31,31), threshold=1e-5):
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
    feature_map1 = feature_map1[:, :, 2:30, 2:30]
    feature_map2 = feature_map2[:,:, 1:24,1:36]
    N, C, H, W = feature_map1.shape
    feature_map1 = torch.nn.functional.interpolate(feature_map1, size=target_size, mode='bilinear')
    feature_map2 = torch.nn.functional.interpolate(feature_map2, size=target_size, mode='bilinear')
    # aggregated_map = feature_map2.mean(dim=1)
    # plt.figure(figsize=(4, 4))
    # plt.imshow(aggregated_map.squeeze().detach().cpu().numpy(), cmap='viridis')
    # plt.title(f'Aggregated Feature Map')
    # plt.axis('off')
    #
    # # 保存图像到指定路径
    # file_path = os.path.join('/data_B/renjie/see', f'sample_.png')
    # plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    # plt.close()  # 关闭图像，避免显示
    # 对每个通道进行全局平均池化来计算通道的激活程度
    activation1 = feature_map1.view(N, C, -1).mean(dim=-1)  # (N, C)
    activation2 = feature_map2.view(N, 256, -1).mean(dim=-1)  # (N, C)
    # activation1 = F.softmax(activation1, dim=1)
    # activation2 = F.softmax(activation2, dim=1)

    # 选择激活程度最高的前 k 个通道
    _, topk_indices1 = torch.topk(activation1, k, dim=1)  # (N, k)
    _, topk_indices2 = torch.topk(activation2, k, dim=1)  # (N, k)

    # 初始化相似度矩阵
    similarities = torch.zeros(N, k)

    for i in range(N):
        # 提取第 i 个样本中激活程度最高的前 k 个通道
        topk_channels1 = feature_map1[i, topk_indices1[i]]  # (k, H, W)
        topk_channels2 = feature_map2[i, topk_indices2[i]]  # (k, H, W)

        # 计算每个通道的余弦相似度
        for j in range(k):
            channel1 = topk_channels1[j].flatten() # 归一化
            channel2 = topk_channels2[j].flatten()  # 归一化
            similarity = F.cosine_similarity(channel1, channel2, dim=0)
            similarities[i, j] = similarity
        # avg_channel1 = topk_channels1.mean(dim=0)  # (H, W)
        # avg_channel2 = topk_channels2.mean(dim=0)  # (H, W)
        #
        # # 可视化并保存前 k 个通道的平均结果
        # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        #
        # # 第一个特征图的平均可视化
        # axes[0].imshow(avg_channel1.detach().cpu().numpy(), cmap='viridis')
        # axes[0].set_title(f'Feat1 Avg of Top {k} Channels')
        # axes[0].axis('off')
        #
        # # 第二个特征图的平均可视化
        # axes[1].imshow(avg_channel2.detach().cpu().numpy(), cmap='viridis')
        # axes[1].set_title(f'Feat2 Avg of Top {k} Channels')
        # axes[1].axis('off')
        #
        # # 保存图像到指定路径
        # file_path = os.path.join('/data_B/renjie/see', f'sample_{i}_avg_top_{k}_channels.png')
        # plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        # plt.close()  # 关闭图像，避免显示

    return torch.mean(similarities)
def main():
    snapshot_path = os.path.join(project_path_, 'pysot/experiments/%s/model.pth' % args.tracker_name)
    config_path = os.path.join(project_path_, 'pysot/experiments/%s/config.yaml' % args.tracker_name)
    # load config
    cfg.merge_from_file(config_path)

    dataset_root = os.path.join(dataset_root_, args.dataset)
    # create model
    '''a model is a Neural Network.(a torch.nn.Module)'''
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, snapshot_path).cuda().eval()

    DiMP = Tracker('tamos', 'tamos_swin_base')
    params = DiMP.get_parameters()
    dimp = DiMP.create_tracker(params)

    # build tracker
    '''a tracker is a object, which consists of not only a NN but also some post-processing'''
    tracker = build_tracker(model)
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    '''GAN'''
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                      True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                                      'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        sim_2=[]
        sim_3=[]
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            print(v_idx)
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    init_info = {'init_bbox': gt_bbox}
                    tracker.init(img, gt_bbox_)
                    dimp.initialize(img, init_info)
                    pred_bbox1 = gt_bbox_
                    pred_bbox2 = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox1)
                else:

                    info = {'previous_output': pred_bbox2}
                    outputs1 = tracker.track(img)
                    outputs2 = dimp.track(img,info)
                    pred_bbox2 = outputs2['target_bbox']
                    feat1=outputs1['feat']  # (31,31)
                    feat2=outputs2['feat']   # (
                    with torch.no_grad():
                        # sim_2.append(compute_top_k_activated_channel_similarity(feat1[0],feat2))
                        sim_3.append(compute_top_k_activated_channel_similarity(feat1[1],feat2))
    # sim_2_stacked = torch.stack(sim_2)
    #
    # 计算堆叠张量的平均值
    # mean_sim2 = torch.mean(sim_2_stacked)
    sim_3_stacked = torch.stack(sim_3)

    # 计算堆叠张量的平均值
    mean_sim3 = torch.mean(sim_3_stacked)
    # print('sim2:',mean_sim2)
    print('sim3',mean_sim3)


if __name__ == '__main__':
    main()
