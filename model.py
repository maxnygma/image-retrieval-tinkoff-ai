import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CosFaceLoss(nn.Module):
    ''' Scale logits to compute LMCL '''

    def __init__(self, in_features, out_features, s=30.0, m=0.6):
        super(CosFaceLoss, self).__init__()

        self.s = s
        self.m = m

        self.ce = nn.CrossEntropyLoss()

        self.weights = nn.Parameter(torch.FloatTensor(out_features, in_features)).to('cuda') # 3D SPACE
        nn.init.xavier_uniform_(self.weights)

    def forward(self, features, labels):
        ''' features - [N, in_features] 
            labels - [1, N]
        '''

        # Dot product
        cosine_sim = F.linear(F.normalize(features, p=2), F.normalize(self.weights, p=2), bias=None)
        delta = cosine_sim - self.m

        # One-hot label encoding
        one_hot = torch.zeros_like(cosine_sim, device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Modify logits
        logits = (one_hot * delta) + ((1.0 - one_hot) * cosine_sim)
        logits *= self.s

        # Compute loss
        loss = self.ce(logits, labels)

        return loss


class ArcFaceLoss(nn.Module):
    ''' Scale logits to compute ArcFace loss '''

    def __init__(self, in_features, out_features, s=30.0, m=0.6, eps=1e-7):
        super(ArcFaceLoss, self).__init__()

        self.s = s
        self.m = m
        self.eps = eps

        self.cos_m = torch.tensor(np.cos(m), device='cuda')
        self.sin_m = torch.tensor(np.sin(m), device='cuda')
        self.theta = torch.tensor(np.cos(math.pi - m), device='cuda')
        self.mm = torch.tensor(np.sin(math.pi - m) * m, device='cuda') # value to keep loss a monotonically decreasing function. But why that way?

        self.ce = nn.CrossEntropyLoss()

        self.weights = nn.Parameter(torch.FloatTensor(out_features, in_features)).to('cuda') 
        nn.init.xavier_uniform_(self.weights)

    def forward(self, features, labels):
        ''' 
            Forward pass

            features - [N, in_features] 
            labels - [1, N]
        '''

        # Dot product
        cosine_sim = F.linear(F.normalize(features, p=2), F.normalize(self.weights, p=2), bias=None)
        
        # Angular margin
        sin_sim = torch.sqrt(1.0 - torch.pow(cosine_sim, 2))

        phi = cosine_sim * self.cos_m - sin_sim * self.sin_m
        phi = torch.where(cosine_sim > self.theta, phi, cosine_sim - self.mm) # check that cosine similarity is not higher than cos(pi - m) to retain decreasing nature

        # # One-hot label encoding
        one_hot = torch.zeros_like(cosine_sim, device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # # Modify logits
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine_sim)
        logits *= self.s

        # Compute loss
        loss = self.ce(logits, labels)

        return loss


class RetrievalModel(nn.Module):
    def __init__(self, backbone, num_classes, is_emb_proj=False):
        super(RetrievalModel, self).__init__()

        self.backbone = backbone
        self.linear = nn.Linear(self.backbone.num_features, num_classes)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.bn = nn.BatchNorm2d(self.backbone.num_features)
        self.bn_1d = nn.BatchNorm1d(num_classes)
        self.dropout = nn.Dropout(0.3)
        # self.prelu = nn.PReLU()

        # Project embeddings to 3D space
        self.is_emb_proj = is_emb_proj
        self.emb_projection = nn.Linear(self.backbone.num_features, 3)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.bn(features)

        features = self.adaptive_avg_pool(features)
        features = features.mean(-1).mean(-1)

        logits = self.linear(features)
        
        if self.is_emb_proj:
            proj_features = self.emb_projection(features)
        else:
            proj_features = None

        return logits, features, proj_features


# https://github.com/ildoonet/pytorch-gradual-warmup-lr
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)