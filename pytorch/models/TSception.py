import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List
from pytorch.models.base_model import base_model
'''
Adapted from: https://github.com/deepBrains/TSception

This is the models of TSception and its variant

To use the models, please manage the data into
the dimension of(mini-batch, 1, EEG-channels,data point)
before feed the data into forward()

For more details about the models, please refer to the paper:
Yi Ding, Neethu Robinson, Qiuhao Zeng, Dou Chen, Aung Aung Phyo Wai, Tih-Shih Lee, Cuntai Guan,
"TSception: A Deep Learning Framework for Emotion Detection Useing EEG"(IJCNN 2020)

Note: The authors use hemispheric spatial kernels in the paper. We use them as inferior vs. superior region kernels.
'''

################################################## TSception ######################################################
class TSception(base_model):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(
            self, 
            num_T, 
            num_S, 
            hidden, 
            num_classes: int = 20, 
            input_size: List = [8, 500], 
            sampling_rate: int = 250, 
            dropout_rate: float = 0.5,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            inception_window = [0.5, 0.25, 0.125],
            one_cycle_lr = True,
            epochs = 100,
            **kwargs
            ):
        # input_size: EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = inception_window #Captures 2Hz, 4Hz, and 8Hz and above respectively
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[-2]), 1), 1, int(self.pool*0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[-2] * 0.5), 1), (int(input_size[-2] * 0.5), 1),
                                         int(self.pool*0.25))
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)

        size = self.get_size(input_size)
        self.fc = nn.Sequential(
            nn.Linear(size[1], hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )
        self.lr = lr
        self.one_cycle_lr = one_cycle_lr
        self.epochs = epochs
        self.ensuredims = Ensure4d()
        self.dimshuffle = Expression(_transpose_to_b_1_c_0) # from [b c t 1] to [b 1 c t] --> first conv over temporal dim with kernel_length
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.ensuredims(x) # from [b c t] to [b c t 1]
        x = self.dimshuffle(x) # from [b c t 1] to [b 1 c t]
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = out.reshape(out.size()[0], -1)
        out = self.fc(out)
        return out

    def get_size(self, input_size):
        # here we use an array with the shape being
        # (1(mini-batch),1(convolutional channel),EEG channel,time data point)
        # to simulate the input data and get the output size
        data = torch.ones((1, 1, input_size[-2], int(input_size[-1])))
        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(data)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        return out.size()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params = self.parameters(), lr = self.lr, weight_decay = self.weight_decay) 
        if self.one_cycle_lr:
            lr = torch.optim.lr_scheduler.OneCycleLR(
                optimizer = optimizer,
                max_lr = self.lr,
                total_steps = len(self.trainer.datamodule.train_dataloader()) * self.epochs,
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
    
######################################### Temporal ########################################
class Tception(base_model):
    def __init__(self, num_classes, input_size, sampling_rate, num_T, hidden, dropout_rate):
        # input_size: channel x datapoint
        super(Tception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125]
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[0]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[1]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[2]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        
        self.BN_t = nn.BatchNorm2d(num_T)

        size = self.get_size(input_size,sampling_rate,num_T)
        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hidden, num_classes))
        
    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out,y),dim = -1)
        y = self.Tception3(x)
        out = torch.cat((out,y),dim = -1)
        out = self.BN_t(out)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
    def get_size(self,input_size,sampling_rate,num_T):
        data = torch.ones((1,1,input_size[0],input_size[1]))        
        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out,y),dim = -1)
        y = self.Tception3(data)
        out = torch.cat((out,y),dim = -1)
        out = self.BN_t(out)
        out = out.view(out.size()[0], -1)
        return out.size()
    
############################################ Spatial ########################################  
class Sception(base_model):
    def __init__(self, num_classes, input_size, sampling_rate, num_S, hidden, dropout_rate):
        # input_size: channel x datapoint
        super(Sception, self).__init__()
        
        self.Sception1 = nn.Sequential(
            nn.Conv2d(1, num_S, kernel_size=(int(input_size[0]),1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(1, num_S, kernel_size=(int(input_size[0]*0.5),1), stride=(int(input_size[0]*0.5),1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))

        self.BN_s = nn.BatchNorm2d(num_S)
        
        size = self.get_size(input_size)
        
        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hidden, num_classes))
        
    def forward(self, x):
        y = self.Sception1(x)
        out = y
        y = self.Sception2(x)
        out = torch.cat((out,y),dim = 2)
        out = self.BN_s(out)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    def get_size(self, input_size):
        data = torch.ones((1,1,input_size[0],input_size[1]))
        y = self.Sception1(data)
        out = y
        y = self.Sception2(data)
        out = torch.cat((out,y),dim = 2)
        out = self.BN_s(out)
        out = out.view(out.size()[0], -1)
        return out.size()
    

#Helper functions (from EEGNet)
def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)

class Expression(nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__ +
            "(expression=%s) " % expression_str
        )

class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x
    
if __name__ == "__main__":
    model = TSception(2,(4,1024),256,9,6,128,0.2)
    #model = Sception(2,(4,1024),256,6,128,0.2)
    #model = Tception(2,(4,1024),256,9,128,0.2)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)