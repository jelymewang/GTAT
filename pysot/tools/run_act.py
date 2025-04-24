# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

from common_path import *
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--tracker_name',default=siam_model_,type=str)
parser.add_argument('--dataset', default= dataset_name_, type=str,
        help='eval one special dataset')
parser.add_argument('--video', default= video_name_, type=str,
        help='eval one special video')
parser.add_argument('--vis', default= False, action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)
'''GAN'''
from GAN_utils_search_1 import *
model_name = opt.model

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
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    '''GAN'''
                    outputs = tracker.track_adv(img,GAN)
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
                        frame_counter = idx + 5 # skip 5 frames
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
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
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
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        act2_t = torch.zeros(512)
        act3_t = torch.zeros(1024)
        act4_t = torch.zeros(2048)
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
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                    act2 = torch.zeros(512)
                    act3 = torch.zeros(1024)
                    act4 = torch.zeros(2048)
                else:
                    # outputs = tracker.track_adv(img,GAN)
                    outputs = tracker.track(img)
                    with torch.no_grad():
                        feat=outputs['feat']
                        act2 = act2 + torch.mean(feat[0], dim=(0,2,3)).cpu()
                        act3 = act3 + torch.mean(feat[1], dim=(0,2,3)).cpu()
                        act4 = act4 + torch.mean(feat[2], dim=(0,2,3)).cpu()
            act2 /= idx + 1
            act3 /= idx + 1
            act4 /= idx + 1
            act2_t += act2
            act3_t += act3
            act4_t += act4
    torch.save(act2_t/180,'act2.pt')
    torch.save(act3_t/180, 'act3.pt')
    torch.save(act4_t/180, 'act4.pt')

if __name__ == '__main__':
    main()
