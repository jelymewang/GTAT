# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os

import torch
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.utils.model_load import load_pretrain
from common_path import project_path_

'''Capsule SiamRPN++(We can use it as one component in higher-level task)'''
class SiamRPNPP():
    def __init__(self,dataset=''):
        if 'OTB' in dataset:
            cfg_file = os.path.join(project_path_,'pysot/experiments/siamrpn_r50_l234_dwxcorr_otb/config.yaml')
            snapshot = os.path.join(project_path_,'pysot/experiments/siamrpn_r50_l234_dwxcorr_otb/model.pth')
        elif 'LT' in dataset:
            cfg_file = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml')
            snapshot = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/model.pth')
        else:
            cfg_file = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
            snapshot = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth')
        # load config
        cfg.merge_from_file(cfg_file)
        # create model
        self.model = ModelBuilder()# A Neural Network.(a torch.nn.Module)
        # load model
        self.model = load_pretrain(self.model, snapshot).cuda().eval()

    def get_heat_map(self, X_crop, softmax=False):
        score_map = self.model.track(X_crop)['cls']#(N,2x5,25,25)
        score_map = score_map.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)#(5HWN,2)
        if softmax:
            score_map = F.softmax(score_map, dim=1).data[:, 1]#(5HWN,)
        return score_map
    def get_cls_reg(self, X_crop, softmax=False,flag=0):
        outputs = self.model.track(X_crop)#(N,2x5,25,25)
        score_map = outputs['cls'].permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)#(5HWN,2)
        reg_res = outputs['loc'].permute(1, 2, 3, 0).contiguous().view(4, -1)
        if softmax:
            score_map = F.softmax(score_map, dim=1).data[:, 1]#(5HWN,)
        feat=outputs['feat'][0]
        activation1 = feat.view(1, 512, -1).mean(dim=-1)  # (N, C)
        if flag==0:
            _, topk_indices1 = torch.topk(activation1, 50, dim=1)  # (N, k)
            self.topk=topk_indices1
        else:
            topk_indices1=self.topk
        activations_topk = feat[0, topk_indices1[0]]
        return activations_topk, score_map, reg_res

    def compute_gradcam_weights_tracking(self, siam_model, target_layer, input_image):
        """
        计算目标跟踪任务中的 Grad-CAM 权重。
        """
        activations = []
        gradients = []

        # 注册前向和后向钩子
        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        # 确保 gradients 是列表
        gradients = []

        # 确保 target_layer 是 nn.Module 对象
        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_backward_hook(backward_hook)

        # 前向传播
        output = siam_model.track(input_image)

        # 计算响应图的和
        response = torch.sum(output['cls'])

        # 反向传播以计算梯度
        siam_model.zero_grad()
        response.backward(retain_graph=True)  # 防止图释放

        # 移除钩子
        handle_forward.remove()
        handle_backward.remove()

        # 获取目标层的特征图和梯度
        activations = activations[0]
        gradientsa = gradients[0]
        # 计算池化梯度（全局平均池化）
        weights = torch.mean(gradientsa, dim=(2, 3), keepdim=True)

        # 计算 Grad-CAM 特征图
        gradcam_map = torch.relu(torch.sum(weights * activations, dim=1)).squeeze()



        score_map = output['cls'].permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)  # (5HWN,2)
        reg_res = output['loc'].permute(1, 2, 3, 0).contiguous().view(4, -1)
        return gradcam_map, score_map, reg_res