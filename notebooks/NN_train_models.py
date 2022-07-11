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

plt.style.use('dark_background')

import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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


# Load data:
out_dir = Path.cwd() / "../input_data/numpy"
out_dir.mkdir(exist_ok=True)
# train_index = np.load(out_dir / "train_indices.npy")
# test_index = np.load(out_dir / "test_indices.npy")
X = np.load(out_dir / "X.npy")
Y = np.load(out_dir / "Y.npy")
stratification_classes = np.load(out_dir / "stratification_classes.npy")

train_val_dataset = TrainValData(X.astype("float32"), Y.astype("float32"))
test_data = TestData(X.astype("float32"))
# %% md


# %%


# EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# %%

import torch.nn as nn


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


# %%

import torch.nn.functional as F

soresen_loss = SorensenDiceLoss()
# soresen_loss = F.binary_cross_entropy_with_logits

# %%


import pytorch_lightning as pl
import torch

NUM_FEAT = 32


class BinaryClassification(pl.LightningModule):
    def __init__(self, learning_rate=LEARNING_RATE, num_feat=NUM_FEAT):
        super(BinaryClassification, self).__init__()

        # Number of input features is 12.
        self.layer_1 = nn.Linear(X.shape[1], num_feat)
        self.layer_2 = nn.Linear(num_feat, num_feat)
        self.layer_out = nn.Linear(num_feat, Y.shape[1])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(num_feat)
        self.batchnorm2 = nn.BatchNorm1d(num_feat)
        self.sigmoid = nn.Sigmoid()

        self.train_f1 = torchmetrics.F1Score()
        self.val_f1 = torchmetrics.F1Score()

        self.save_hyperparameters()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.layer_out(x)
        x = self.sigmoid(x)

        # for i in range(20):
        #     self.log("prediction_{}".format(i), x[:,i].mean())
        return x

    def predict_step(self, batch, batch_idx):
        return self.forward(batch[0]), batch[1]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = soresen_loss(y_hat, y)
        self.train_f1(y_hat, y.int())
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = soresen_loss(y_hat, y)
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

    def train_dataloader(self):
        train_loader = DataLoader(dataset=train_val_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(dataset=train_val_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
        return val_loader


# %%

from pytorch_lightning.callbacks import LearningRateMonitor


# Do stratification and cross-val-loop:

NUM_SPLITS = 10
skf = sklearn.model_selection.StratifiedKFold(n_splits=NUM_SPLITS)
skf.get_n_splits()


pbar_cross_split = tqdm(skf.split(range(X.shape[0]), stratification_classes),
                                leave=False, total=NUM_SPLITS)

all_results = pd.DataFrame()

for fold, (train_index, test_index) in enumerate(pbar_cross_split):
    # Using PyTorch sampler:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_index.tolist())
    valid_sampler = torch.utils.data.SubsetRandomSampler(test_index.tolist())

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = BinaryClassification()

    # train model
    trainer = pl.Trainer(
        callbacks=[lr_monitor],
        default_root_dir=Path.cwd() / "../training_data_torch",
        gradient_clip_algorithm="norm",
        enable_progress_bar=True,
        max_epochs=300,
        detect_anomaly=True,
        #    auto_lr_find=True,
        # ckpt_path="path",
    )

    trainer.fit(model=model)

    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, sampler=valid_sampler)

    prediction = trainer.predict(model, dataloaders=test_loader)

    # Save predictions:
    nb_batches = len(prediction)
    pred_array = np.concatenate([tensor[0].numpy() for tensor in prediction])
    pred_indices = np.concatenate([tensor[1].numpy() for tensor in prediction])

    lc_results = pd.DataFrame(pred_array, index=pred_indices)
    lc_results["fold"] = fold
    all_results = pd.concat([all_results, lc_results])

all_results.to_csv(Path.cwd() / "../results/nn_results.csv")

