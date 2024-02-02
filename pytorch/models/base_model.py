import torch
import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import elu
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

class base_model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=20)
        self.config = None
        self.save_hyperparameters()
        self.one_cycle_lr = True
        self.predictions = []
        self.ground_truth = []

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(params = self.parameters(), lr = self.config["lr"], weight_decay = self.config["weight_decay"]) 
    #     if self.one_cycle_lr:
    #         lr = torch.optim.lr_scheduler.OneCycleLR(
    #             optimizer = optimizer,
    #             max_lr = self.config["lr"],
    #             epochs = self.config["epochs"],
    #             steps_per_epoch = self.trainer.estimated_stepping_batches // self.config["epochs"],
    #             cycle_momentum = True
    #             )
            
    #         lr_scheduler = {
    #             "scheduler": lr,
    #             "interval": "step",
    #             "name": "Learning Rate Scheduling"
    #         }
    #         return [optimizer], [lr_scheduler]
        # else:
        #     return [optimizer]

    def training_step(self, batch, batch_idx):
        x,y = batch
        logit = self.forward(x.float()) 
        train_loss = self.loss(logit, y)
        _, y_pred = torch.max(logit, dim = 1)
        self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.acc(y_pred, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return train_loss
        
    def validation_step(self, batch, batch_idx):
        x,y = batch
        logit = self.forward(x.float()) 
        val_loss = self.loss(logit, y)
        _, y_pred = torch.max(logit, dim = 1)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.acc(y_pred, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x,y = batch
        print("TEST")
        logit = self.forward(x.float())
        test_loss = self.loss(logit, y)
        _, y_pred = torch.max(logit, dim = 1)
        self.predictions.append(y_pred)
        self.ground_truth.append(y)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", self.acc(y_pred, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return test_loss