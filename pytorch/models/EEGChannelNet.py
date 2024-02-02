"""
# This code is used for the EEG ChannelNet and is based on code from the following source:
# Title: EEG Visual Classification
# Author: PerceiveLab
# URL: https://github.com/perceivelab/eeg_visual_classification/blob/main/models/EEGChannelNet.py
# Date Accessed: 23.07.2023
"""
# This is the model presented in the work: S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, M. Shah, Decoding Brain Representations by 
# Multimodal Learning of Neural Activity and Visual Features,  IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2020, doi: 10.1109/TPAMI.2020.2995909
# Note the model is adapted to work with the general pipeline used in this study (pytorch lightning and wandb for logging, and reduction of input channels)
# Additionally the optimizer was integrated into the model with the default values found in the authors' respository

from typing import Any
import torch
import torch.nn as nn

from pytorch.models.layers import * 
from pytorch.models.base_model import base_model

class FeaturesExtractor(nn.Module):
    def __init__(self, in_channels, temp_channels, out_channels, input_width, in_height,
                 temporal_kernel, temporal_stride, temporal_dilation_list, num_temporal_layers,
                 num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride,
                 dropout):
        super().__init__()

        self.temporal_block = TemporalBlock(
            in_channels, temp_channels, num_temporal_layers, temporal_kernel, temporal_stride, temporal_dilation_list, input_width, dropout
        )

        self.spatial_block = SpatialBlock(
            temp_channels * num_temporal_layers, out_channels, num_spatial_layers, spatial_stride, in_height, dropout
        )

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(
                    out_channels * num_spatial_layers, out_channels * num_spatial_layers
                ),
                ConvLayer2D(
                    out_channels * num_spatial_layers, out_channels * num_spatial_layers, down_kernel, down_stride, 0, 1, dropout
                )
            ) for i in range(num_residual_blocks)
        ])

        self.final_conv = ConvLayer2D(
            out_channels * num_spatial_layers, out_channels, down_kernel, 1, 0, 1, dropout
        )

    def forward(self, x):
        out = self.temporal_block(x) #1x40x8x250
        out = self.spatial_block(out) #1x150x4x250
        if len(self.res_blocks) > 0:
            for res_block in self.res_blocks:
                out = res_block(out)
        out = self.final_conv(out)
        return out

class ChannelNet(base_model):
    '''The model for EEG classification.
    The input is a tensor where each row is a channel the recorded signal and each colums is a time sample.
    The model performs different 2D to extract temporal e spatial information.
    The output is a vector of classes where the maximum value is the predicted class.
    Args:
        in_channels: number of input channels
        temp_channels: number of features of temporal block
        out_channels: number of features before classification
        num_classes: number possible classes
        embedding_size: size of the embedding vector
        input_width: width of the input tensor (necessary to compute classifier input size)
        input_height: height of the input tensor (necessary to compute classifier input size)
        temporal_dilation_list: list of dilations for temporal convolutions, second term must be even
        temporal_kernel: size of the temporal kernel, second term must be even (default: (1, 32))
        temporal_stride: size of the temporal stride, control temporal output size (default: (1, 2))
        num_temp_layers: number of temporal block layers
        num_spatial_layers: number of spatial layers
        spatial_stride: size of the spatial stride
        num_residual_blocks: the number of residual blocks
        down_kernel: size of the bottleneck kernel
        down_stride: size of the bottleneck stride
        '''
    def __init__(self, in_channels=1, temp_channels=10, out_channels=50, num_classes=20, embedding_size=1000,
                 input_width=500, input_height=8, temporal_dilation_list=[(1,1),(1,2),(1,4),(1,8),(1,16)],
                 temporal_kernel=(1,33), temporal_stride=(1,2),num_temp_layers=4,num_spatial_layers=4, 
                 spatial_stride=(2,1), num_residual_blocks=2, down_kernel=(1,3), down_stride=2,
                 dropout = 0.5, lr = 0.001, one_cycle_lr = False, epochs = 100, **kwargs):
        super().__init__()
        self.lr = lr
        self.one_cycle_lr = one_cycle_lr
        self.epochs = epochs
        self.encoder = FeaturesExtractor(in_channels, temp_channels, out_channels, input_width, input_height,
                                     temporal_kernel, temporal_stride,temporal_dilation_list, num_temp_layers,
                                     num_spatial_layers, spatial_stride, num_residual_blocks, down_kernel, down_stride,
                                     dropout
                                     )

        encoding_size = self.encoder(torch.zeros(1, in_channels, input_height, input_width)).contiguous().view(-1).size()[0]

        self.classifier = nn.Sequential(
            nn.Linear(encoding_size, num_classes),
            # nn.ReLU(True),
            # nn.Linear(embedding_size, num_classes), 
        )

    def forward(self, x):
        out = self.encoder(x.unsqueeze(1))
        out = out.view(x.size(0), -1) #bsx50x3x59 -> bsx8850
        out = self.classifier(out) #bsx8850 -> bsx1000 -> bsx20
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params = self.parameters(), lr = self.lr)#, weight_decay = self.weight_decay) 
        if self.one_cycle_lr:
            lr = torch.optim.lr_scheduler.OneCycleLR(
                optimizer = optimizer,
                max_lr = self.lr,
                total_steps = len(self.trainer.datamodule.train_dataloader()) * self.epochs,
                # epochs = self.epochs,
                # steps_per_epoch = self.trainer.estimated_stepping_batches // self.epochs,
                cycle_momentum = True
                )
            
            lr_scheduler = {
                "scheduler": lr,
                "interval": "step",
                "name": "Learning Rate Scheduling"
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]