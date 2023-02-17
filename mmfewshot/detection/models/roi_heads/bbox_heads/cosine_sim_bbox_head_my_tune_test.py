# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import ConvFCBBoxHeadMyTune
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import pickle
import numpy as np

@HEADS.register_module()
class CosineSimBBoxHeadMyTuneTest(ConvFCBBoxHeadMyTune):
    """BBOxHead for `TFA <https://arxiv.org/abs/2003.06957>`_.

    The code is modified from the official implementation
    https://github.com/ucbdrive/few-shot-object-detection/

    Args:
        scale (int): Scaling factor of `cls_score`. Default: 20.
        learnable_scale (bool): Learnable global scaling factor.
            Default: False.
        eps (float): Constant variable to avoid division by zero.
    """

    def __init__(self,
                 scale: int = 20,
                 learnable_scale: bool = False,
                 eps: float = 1e-5,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # override the fc_cls in :obj:`ConvFCBBoxHead`
        if self.with_cls:
            self.fc_cls = nn.Linear(
                self.cls_last_dim, self.num_classes + 1, bias=False)

        # learnable global scaling factor
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1) * scale)
        else:
            self.scale = scale
        self.eps = eps
        
        # base class statistics
        f = open('/data/zhaozhiyuan/mmfewshot-main/base_statistics/split3/2022-03-05/10:38:58/base_statistics.pt', 'rb')
        self.base_statistics = pickle.load(f)
        f.close()
        self.phi = 0.5
        self.k = 5
        self.alpha = 0.2
        self.new_feat_num = 1
        
        self.mean_all_base = []
        for k, v in self.base_statistics.items():
            self.mean_all_base.append(torch.Tensor(v['mean']).unsqueeze(0))
        self.mean_all_base = torch.cat(self.mean_all_base, dim=0).cuda()
        self.var_all_base = []
        for k, v in self.base_statistics.items():
            self.var_all_base.append(torch.Tensor(v['variance']).unsqueeze(0))
        self.var_all_base = torch.cat(self.var_all_base, dim=0).cuda()

        self.branch1 = nn.Sequential(
            nn.Linear(self.mean_all_base.shape[1]*2, self.mean_all_base.shape[1]*2),
            nn.ReLU(),
            nn.Linear(self.mean_all_base.shape[1]*2, self.mean_all_base.shape[1]),
            nn.ReLU(),
            nn.Linear(self.mean_all_base.shape[1], self.mean_all_base.shape[1]),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Linear(self.mean_all_base.shape[0], self.mean_all_base.shape[0]//2),
            nn.Linear(self.mean_all_base.shape[0]//2, self.mean_all_base.shape[0]//4),
            nn.Linear(self.mean_all_base.shape[0]//4, self.mean_all_base.shape[0]//8),
            nn.Linear(self.mean_all_base.shape[0]//8, 1)
        )
        
    def transform(self, embed, label):
        if label >= 15 and label <= 19:
            new_samples_all = []
            for i in range(self.new_feat_num):
                new_sample_base_all = []
                for k in range(self.mean_all_base.shape[0]):
                    m = torch.distributions.multivariate_normal.MultivariateNormal(self.mean_all_base[i], scale_tril=self.var_all_base[i])
                    new_sample_base = m.sample()
                    new_sample_base = torch.cat([new_sample_base, embed], dim=0).unsqueeze(0)
                    new_sample_base_all.append(new_sample_base)
                new_sample_base_all = torch.cat(new_sample_base_all, dim=0)
            
                # step1
                s1 = self.branch1(new_sample_base_all)
                # step2
                s2 = s1.permute(1,0)
                s2 = self.branch2(s2)
                s2 = s2.squeeze(1)
                new_sample_novel = s2 + embed
                new_sample_novel = new_sample_novel.unsqueeze(0)
                new_samples_all.append(new_sample_novel)
            new_samples_all = torch.cat(new_samples_all, dim=0).to(embed.device)
            
        else:
            m = torch.distributions.multivariate_normal.MultivariateNormal(self.mean_all_base[label], scale_tril=self.var_all_base[label])
            new_samples_all = []
            for i in range(self.new_feat_num):
                new_sample = m.sample().unsqueeze(0)
                new_samples_all.append(new_sample)
            new_samples_all = torch.cat(new_samples_all, dim=0).to(embed.device)
        
        return new_samples_all
        
    def tukey_ladder(self, x, labels):
        for i, label in enumerate(labels):
            if i >= 15 and i <= 19:
                x[i] = x[i] ** self.phi
        return x
        
    def forward(self, x: Tensor, labels = None) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            x (Tensor): Shape of (num_proposals, C, H, W).

        Returns:
            tuple:
                cls_score (Tensor): Cls scores, has shape
                    (num_proposals, num_classes).
                bbox_pred (Tensor): Box energies / deltas, has shape
                    (num_proposals, 4).
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
                
        """
        # tukey's lader
        if self.training:
            if labels is not None:
                for i, label in enumerate(labels):
                    if i >= 15 and i <= 19:
                        x[i] = x[i] ** self.phi
        """
            
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        if x_cls.dim() > 2:
            x_cls = torch.flatten(x_cls, start_dim=1)
        
        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x_cls, p=2, dim=1).unsqueeze(1).expand_as(x)
        
        x_cls_normalized = x_cls.div(x_norm + self.eps)
        # normalize weight
        with torch.no_grad():
            temp_norm = torch.norm(
                self.fc_cls.weight, p=2,
                dim=1).unsqueeze(1).expand_as(self.fc_cls.weight)
            self.fc_cls.weight.div_(temp_norm + self.eps)
        # calculate and scale cls_score
        cls_score = self.scale * self.fc_cls(
            x_cls_normalized) if self.with_cls else None

        """
        count = 0
        new_sample_index = []
        for i, label in enumerate(labels):
            if label >=15 and label <= 19:
                new_sample_index.append(i)
                count += 1
        if len(new_sample_index) != 0:
            x_reg_new = torch.rand(len(new_sample_index)*self.new_feat_num,1024).to(x.device)
            bbox_pred_new = self.fc_reg(x_reg_new) if self.with_reg else None
            x_cls_normalized_new = torch.rand(len(new_sample_index)*self.new_feat_num,1024).to(x.device)
            cls_score_new = self.scale * self.fc_cls(
                x_cls_normalized_new) if self.with_cls else None
        """
        """
        transfer statistics: base->novel
        """
        cls_score_new, bbox_pred_new, new_sample_index = None, None, None
        #print(labels)
        if self.training and labels is not None:
            new_sample_index = []
            new_sample_all = []
            
            if labels is not None:
                for i, label in enumerate(labels):
                    #if label >= 15 and label <= 19:
                    #if i == 0:
                    if label != 20:
                        # transform
                        novel_embed = x[i]
                        new_sample = self.transform(novel_embed, label)
                        #new_sample = torch.rand(self.new_feat_num, 1024).to(x.device)
                        new_sample_index.append(i)
                        new_sample_all.append(new_sample)
            
                        
            # 有新生成的sample
            if len(new_sample_index) != 0:
                new_sample_all = torch.cat(new_sample_all, dim = 0)
                #new_sample_all = torch.rand(len(new_sample_index),1024).to(x.device)
                #print(new_sample_all.shape)
            
                # predict for new samples
                # separate branches
                x_cls_new = new_sample_all
                x_reg_new = new_sample_all
                #x_cls_new = torch.rand(len(new_sample_index),1024).to(x.device)
                #x_reg_new = torch.rand(len(new_sample_index),1024).to(x.device)
                
                for conv in self.cls_convs:
                    x_cls_new = conv(x_cls_new)
                if x_cls_new.dim() > 2:
                    if self.with_avg_pool:
                        x_cls_new = self.avg_pool(x_cls_new)
                    x_cls_new = x_cls_new.flatten(1)
                for fc in self.cls_fcs:
                    x_cls_new = self.relu(fc(x_cls_new))

                for conv in self.reg_convs:
                    x_reg_new = conv(x_reg_new)
                if x_reg_new.dim() > 2:
                    if self.with_avg_pool:
                        x_reg_new = self.avg_pool(x_reg_new)
                    x_reg_new = x_reg_new.flatten(1)
                for fc in self.reg_fcs:
                    x_reg_new = self.relu(fc(x_reg_new))
            
                # bbox pred
                #_x_reg_new = torch.rand(len(new_sample_index)*self.new_feat_num,1024).to(x.device)
                bbox_pred_new = self.fc_reg(x_reg_new) if self.with_reg else None
            
                # classification
                if x_cls_new.dim() > 2:
                    x_cls_new = torch.flatten(x_cls_new, start_dim=1)
        
                # normalize the input x along the `input_size` dimension
                x_norm_new = torch.norm(x_cls_new, p=2, dim=1).unsqueeze(1).expand_as(new_sample_all)
            
                x_cls_normalized_new = x_cls_new.div(x_norm_new + self.eps)
                #x_cls_normalized_new = torch.rand(len(new_sample_index)*self.new_feat_num,1024).to(x.device)
                """
                # normalize weight
                with torch.no_grad():
                    temp_norm_new = torch.norm(
                        self.fc_cls.weight, p=2,
                        dim=1).unsqueeze(1).expand_as(self.fc_cls.weight)
                    self.fc_cls.weight.div_(temp_norm_new + self.eps)
                # calculate and scale cls_score
                """
                cls_score_new = self.scale * self.fc_cls(
                    x_cls_normalized_new) if self.with_cls else None
        
        
        return cls_score, bbox_pred, cls_score_new, bbox_pred_new, new_sample_index
        #return cls_score, bbox_pred, None, None, new_sample_index
