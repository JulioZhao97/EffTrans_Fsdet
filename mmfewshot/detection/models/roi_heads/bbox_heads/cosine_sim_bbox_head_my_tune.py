# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import ConvFCBBoxHeadMyTune
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable

import os
import sys
import pickle
import numpy as np

@HEADS.register_module()
class CosineSimBBoxHeadMyTune(ConvFCBBoxHeadMyTune):
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
                 k: int = 15,
                 alpha : float = 0.1,
                 repeat : int = 1,
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
        f = open('/data/zhaozhiyuan/mmfewshot-main/base_statistics/voc/split1/base_statistics.pt', 'rb')
        self.base_statistics = pickle.load(f)
        f.close()
        self.phi = 0.5
        self.k = k
        self.alpha = alpha
        self.repeat = repeat
        print('k={} alpha={} repeat={}'.format(self.k, self.alpha, self.repeat))
        self.new_feat_num = 1
        
        self.mean_all_base = []
        for k, v in self.base_statistics.items():
            self.mean_all_base.append(torch.Tensor(v['mean']).unsqueeze(0))
        self.mean_all_base = torch.cat(self.mean_all_base, dim=0).cuda()
        
        # tsne
        # self.iter = 0
        
    """
    def transform(self, embed):
        # select topk classes
        dist_all = [None for i in range(15)]
        for c, statistic in self.base_statistics.items():
            dist = F.pairwise_distance(embed.unsqueeze(0), torch.Tensor(statistic['mean']).to(embed.device).unsqueeze(0))
            dist_all[int(c)] = dist[0]
        dist_all = torch.Tensor(dist_all)
        topk = torch.topk(dist_all, self.k, largest = False)[1]
        
        #print(embed.shape, self.mean_all_base.shape)
        #sys.exit()
        
        # distribution calibration
        mean = np.zeros(embed.shape[0])
        variance = np.zeros((embed.shape[0], embed.shape[0]))
        for c in topk:
            mean += self.base_statistics[str(c.item())]['mean']
            variance += self.base_statistics[str(c.item())]['variance']
        cal_mean = (mean + embed.cpu().numpy()) / (self.k + 1)
        cal_var = variance / self.k + self.alpha
        
        
        # sample new features
        #new_sample = np.random.multivariate_normal(cal_mean, cal_var, self.new_feat_num)
        #new_sample = np.random.multivariate_normal(np.random.rand(1024), np.random.rand(1024,1024), self.new_feat_num)
        #new_sample = torch.Tensor(new_sample).to(embed.device)
        #new_sample = torch.rand(self.new_feat_num, 1024).to(embed.device)
        #new_sample = Variable(new_sample, requires_grad = True)
        
        cal_mean, cal_var = torch.Tensor(cal_mean), torch.Tensor(cal_var)
        #print(cal_mean.shape, cal_var.shape)
        m = torch.distributions.multivariate_normal.MultivariateNormal(cal_mean, scale_tril=cal_var)
        new_samples_all = []
        for i in range(self.new_feat_num):
            new_sample = m.sample().unsqueeze(0)
            # filter distance
            #dist = torch.nn.functional.pairwise_distance(new_sample.to(embed.device), embed.unsqueeze(0))
            #if dist[0] >= 20:
            #    continue
            new_samples_all.append(new_sample)
        if len(new_samples_all) > 0:
            new_samples_all = torch.cat(new_samples_all, dim=0).to(embed.device)
        else:
            new_samples_all = None
        
        return new_samples_all
    """
    
    def transform(self, embed, label, index):
        if label >= 15 and label <= 19:
            # select topk classes
            dist_all = [None for i in range(15)]
            for c, statistic in self.base_statistics.items():
                dist = F.pairwise_distance(embed.unsqueeze(0), torch.Tensor(statistic['mean']).to(embed.device).unsqueeze(0))
                dist_all[int(c)] = dist[0]
            dist_all = torch.Tensor(dist_all)
            topk = torch.topk(dist_all, self.k, largest = False)[1]
        
            # distribution calibration
            mean = np.zeros(embed.shape[0])
            variance = np.zeros((embed.shape[0], embed.shape[0]))
            for c in topk:
                mean += self.base_statistics[str(c.item())]['mean']
                variance += self.base_statistics[str(c.item())]['variance']
            cal_mean = (mean + embed.cpu().numpy()) / (self.k + 1)
            cal_var = variance / self.k + self.alpha
        
            cal_mean, cal_var = torch.Tensor(cal_mean), torch.Tensor(cal_var)
            #print(cal_mean.shape, cal_var.shape)
            m = torch.distributions.multivariate_normal.MultivariateNormal(cal_mean, scale_tril=cal_var)
            new_samples_all = []
            for i in range(self.new_feat_num):
                min_dis, min_sample = None, None
                for j in range(self.repeat):
                    new_sample = m.sample().unsqueeze(0)
                    # L2-distance
                    d = torch.nn.functional.pairwise_distance(new_sample.cuda(), embed.unsqueeze(0))
                    if min_dis is None:
                        min_dis = d
                        min_sample = new_sample
                    else:
                        if d < min_dis:
                            min_dis = d
                            min_sample = new_sample
                new_samples_all.append(min_sample) 
            if len(new_samples_all) > 0:
                new_samples_all = torch.cat(new_samples_all, dim=0).to(embed.device)
            else:
                new_samples_all = None
        else:
            c = label.cpu().item()
            cal_mean, cal_var = self.base_statistics[str(c)]['mean'], self.base_statistics[str(c)]['variance']
            cal_mean, cal_var = torch.Tensor(cal_mean), torch.Tensor(cal_var)
            m = torch.distributions.multivariate_normal.MultivariateNormal(cal_mean, scale_tril=cal_var)
            new_samples_all = []
            for i in range(self.new_feat_num):
                min_dis, min_sample = None, None
                for j in range(self.repeat):
                    new_sample = m.sample().unsqueeze(0)
                    # L2-distance
                    d = torch.nn.functional.pairwise_distance(new_sample.cuda(), embed.unsqueeze(0))
                    if min_dis is None:
                        min_dis = d
                        min_sample = new_sample
                    else:
                        if d < min_dis:
                            min_dis = d
                            min_sample = new_sample
                new_samples_all.append(min_sample) 
            new_samples_all = torch.cat(new_samples_all, dim=0).to(embed.device)
            
            
        # save for tsne
        """
        if self.training and self.iter >= 4000:
            #print('save')
            #print('{}/{}_{}'.format(label.item(),self.iter,index))
            #original
            if not os.path.exists('base_statistics/tsne/ori/{}'.format(label.item())):
                os.mkdir('base_statistics/tsne/ori/{}'.format(label.item()))
            f = open('base_statistics/tsne/ori/{}/{}_{}.pt'.format(label.item(),self.iter,index), 'wb')
            pickle.dump(embed.cpu().numpy(), f)
            f.close()
            # new
            if not os.path.exists('base_statistics/tsne/new/{}'.format(label.item())):
                os.mkdir('base_statistics/tsne/new/{}'.format(label.item()))
            f = open('base_statistics/tsne/new/{}/{}_{}.pt'.format(label.item(),self.iter,index), 'wb')
            pickle.dump(new_samples_all[0].cpu().numpy(), f)
            f.close()
        """
        
        return new_samples_all
        
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
        
        """
        # tsne
        if self.training:
            self.iter += 1
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
        only using in split1
        """
        # tukey's lader
        if self.training:
            if labels is not None:
                for i, label in enumerate(labels):
                    if label >= 15 and label <= 19:
                        x[i] = x[i] ** self.phi
        
            
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
                        new_sample = self.transform(novel_embed, label, i)
                        # filter distance
                        if new_sample is not None:
                            new_sample_index.append(i)
                            new_sample_all.append(new_sample)
            
                        
            # 有新生成的sample
            if len(new_sample_index) != 0:
                new_sample_all = torch.cat(new_sample_all, dim = 0)
                
                # predict for new samples
                # separate branches
                x_cls_new = new_sample_all
                x_reg_new = new_sample_all
                
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
                bbox_pred_new = self.fc_reg(x_reg_new) if self.with_reg else None
            
                # classification
                if x_cls_new.dim() > 2:
                    x_cls_new = torch.flatten(x_cls_new, start_dim=1)
        
                # normalize the input x along the `input_size` dimension
                x_norm_new = torch.norm(x_cls_new, p=2, dim=1).unsqueeze(1).expand_as(new_sample_all)
            
                x_cls_normalized_new = x_cls_new.div(x_norm_new + self.eps)
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
        #return cls_score, bbox_pred, cls_score_new, None, new_sample_index
