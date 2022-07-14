# %%
import torchmetrics
from sklearn.model_selection import train_test_split
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.model_selection
from matplotlib import pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor



## train data
class TrainValData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index], index

    def __len__(self):
        return len(self.X_data)


# %%


def flatten_samples(input_):
    """
    Flattens a tensor or a variable such that the channel axis is first and the sample axis
    is second. The shapes are transformed as follows:
        (N, C, H, W) --> (C, N * H * W)
        (N, C, D, H, W) --> (C, N * D * H * W)
        (N, C) --> (C, N)
    The input must be atleast 2d.
    """
    assert input_.dim() >= 2, "Tensor or variable must be atleast 2D. Got one of dim {}.".format(input_.dim())
    # Get number of channels
    num_channels = input_.size(1)
    # Permute the channel axis to first
    permute_axes = list(range(input_.dim()))
    permute_axes[0], permute_axes[1] = permute_axes[1], permute_axes[0]
    # For input shape (say) NCHW, this should have the shape CNHW
    permuted = input_.permute(*permute_axes).contiguous()
    # Now flatten out all but the first axis and return
    flattened = permuted.view(num_channels, -1)
    return flattened


class SorensenDiceLoss(nn.Module):
    """
    Computes a loss scalar, which when minimized maximizes the Sorensen-Dice similarity
    between the input and the target. For both inputs and targets it must be the case that
    `input_or_target.size(1) = num_channels`.
    """

    def __init__(self, weight=None, channelwise=True, eps=1e-6):
        """
        Parameters
        ----------
        weight : torch.FloatTensor or torch.cuda.FloatTensor
            Class weights. Applies only if `channelwise = True`.
        channelwise : bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        """
        super(SorensenDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.channelwise = channelwise
        self.eps = eps

    def forward(self, input, target):
        """
        input:      torch.FloatTensor or torch.cuda.FloatTensor
        target:     torch.FloatTensor or torch.cuda.FloatTensor

        Expected shape of the inputs: (batch_size, nb_channels, ...)
        """
        assert input.size() == target.size()
        if not self.channelwise:
            numerator = (input * target).sum()
            denominator = (input * input).sum() + (target * target).sum()
            loss = -2. * (numerator / denominator.clamp(min=self.eps))
        else:
            # TODO This should be compatible with Pytorch 0.2, but check
            # Flatten input and target to have the shape (C, N),
            # where N is the number of samples
            input = flatten_samples(input)
            target = flatten_samples(target)
            # Compute numerator and denominator (by summing over samples and
            # leaving the channels intact)
            numerator = (input * target).sum(-1)
            denominator = (input * input).sum(-1) + (target * target).sum(-1)
            channelwise_loss = -2 * (numerator / denominator.clamp(min=self.eps))
            if self.weight is not None:
                # With pytorch < 0.2, channelwise_loss.size = (C, 1).
                if channelwise_loss.dim() == 2:
                    channelwise_loss = channelwise_loss.squeeze(1)
                assert self.weight.size() == channelwise_loss.size()
                # Apply weight
                channelwise_loss = self.weight * channelwise_loss
            # Sum over the channels to compute the total loss
            loss = channelwise_loss.sum()
        return loss


class SimpleTwoLayersNN(pl.LightningModule):
    def __init__(self,
                 num_feat,
                 nb_in_feat,
                 nb_out_feat,
                 loss,
                 final_activation=None,
                 learning_rate=0.001):
        super(SimpleTwoLayersNN, self).__init__()

        # Number of input features is 12.
        self.layer_1 = nn.Linear(nb_in_feat, num_feat)
        self.layer_2 = nn.Linear(num_feat, num_feat)
        self.layer_out = nn.Linear(num_feat, nb_out_feat)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(num_feat)
        self.batchnorm2 = nn.BatchNorm1d(num_feat)
        self.final_activation = final_activation

        self.train_f1 = torchmetrics.F1Score()
        self.val_f1 = torchmetrics.F1Score()

        self.loss = loss

        self.save_hyperparameters()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.layer_out(x)
        if self.final_activation is not None:
            x = self.final_activation(x)

        # for i in range(20):
        #     self.log("prediction_{}".format(i), x[:,i].mean())
        return x

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch[0]), batch[1]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.train_f1(y_hat, y.int())
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        self.val_f1(y_hat, y.int())
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        # metric = torchmetrics.F1Score(multiclass=False, average='macro', num_classes=2)
        # metric(y_hat, y.int())

        self.log("val_loss", val_loss)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     test_loss = F.binary_cross_entropy_with_logits(y_hat, y)
    #     self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
        return [optimizer], [lr_scheduler]

    # def train_dataloader(self):
    #     train_loader = DataLoader(dataset=train_val_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    #     return train_loader
    #
    # def val_dataloader(self):
    #     val_loader = DataLoader(dataset=train_val_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
    #     return val_loader

