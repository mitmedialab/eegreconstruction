"""
Note: this code is adapted from https://github.com/eeyhsong/EEG-Conformer/tree/main
The initial Conformer model is adapted to be used with the pipeline to do hyperparameter search.
"""
# remember to change paths
import os
#gpus = [0]
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

from torch.nn.functional import elu
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from pytorch.models.base_model import base_model

# Convolution module
# use conv to capture local features, instead of postion embedding.

class PatchEmbedding(nn.Module):
    def __init__(self, conv_depth = 40, temporal_conv = 25, temporal_pool = 60, emb_size=40, dropout=0.5):
        # self.patch_size = patch_size
        super().__init__()
        temporal_stride = temporal_pool // 4
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, conv_depth, (1, 25), (1, 1)),
            nn.Conv2d(conv_depth, conv_depth, (8, 1), (1, 1)), #(22,1) changed to (8,1)
            nn.BatchNorm2d(conv_depth),
            nn.ELU(),
            nn.AvgPool2d((1, temporal_pool), (1, temporal_stride)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(conv_depth, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=2,
                 drop_p=0.5,
                 forward_expansion=4
                 ):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, num_heads, forward_expansion, dropout):
        super().__init__(*[TransformerEncoderBlock(
            emb_size = emb_size, 
            num_heads = num_heads, 
            drop_p = dropout,
            forward_expansion = forward_expansion,
            ) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes, dropout):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(out_features = 256), #replaced Linear with LazyLinear for hyperparam search
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 20)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out

class Conformer(base_model):
    def __init__(
            self, 
            #General
            n_classes=20, 
            lr = 0.0002,
            epochs = 100,
            one_cycle_lr = False,
            warmup_cosine_annealing = True,
            warmup_epochs = 40,

            #Convolutional Module
            temporal_conv = 25,
            temporal_pool = 60,
            forward_expansion = 1, 
            dropout_conv = 0.5,
            emb_size=40,
            #Attention Module 
            depth=6, 
            num_heads = 10,
            conv_depth = 40,
            dropout_transformer = 0.5,
            #Classifier Module
            dropout_classifier = 0.3,
            **kwargs):
        super().__init__()
        self.model = nn.Sequential(
            PatchEmbedding(
                conv_depth = conv_depth,
                temporal_conv = temporal_conv,
                temporal_pool = temporal_pool,
                emb_size = emb_size,
                dropout = dropout_conv,
                ),
            TransformerEncoder(
                depth = depth,
                emb_size = emb_size,
                num_heads = num_heads,
                dropout = dropout_transformer,
                forward_expansion = forward_expansion,
                ),
            ClassificationHead(
                emb_size = emb_size, 
                n_classes = n_classes,
                dropout = dropout_classifier
                )
        )
        self.lr = lr
        self.epochs = epochs
        self.one_cycle_lr = one_cycle_lr
        self.warmup_cosine_annealing = warmup_cosine_annealing
        self.warmup_epochs = warmup_epochs

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.model(x)
        return x
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params = self.model.parameters(), lr = self.lr, betas = (0.5, 0.999))
        if self.one_cycle_lr:
            one_cycle = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer = optimizer,
                    max_lr = self.lr,
                    total_steps = len(self.trainer.datamodule.train_dataloader()) * self.epochs,
                    cycle_momentum = True
                    )
            
            lr_scheduler = {
                "scheduler": one_cycle, #lr
                "interval": "step",
                "name": "Learning Rate Scheduling"
            }
            return [optimizer], [lr_scheduler]
        elif self.warmup_cosine_annealing:
            optimizer = torch.optim.AdamW(params = self.model.parameters(), lr = self.lr, betas = (0.9, 0.999))
            # torch_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.epochs-self.warmup_epochs)
            # warmup_cosine_annealing = create_lr_scheduler_with_warmup(
            #     torch_lr_scheduler, 
            #     warmup_start_value = self.lr, 
            #     warmup_end_value = self.lr*4, 
            #     warmup_duration = self.warmup_epochs
            #     ) 
            warmup_cosine_annealing = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs = self.warmup_epochs, max_epochs = self.epochs)
            lr_scheduler = {
                "scheduler": warmup_cosine_annealing, #lr
                "interval": "step",
                "name": "Learning Rate Scheduling"
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]
        
    # def lr_scheduler_step(self, scheduler):
    #     scheduler.step(epoch=self.current_epoch)