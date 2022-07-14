from pathlib import Path

import pandas as pd
import numpy as np
import sklearn
import tqdm

import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from pred_spot_intensity.pytorch_utils import TrainValData, TestData, SorensenDiceLoss, SimpleTwoLayersNN
from pred_spot_intensity.sklearn_training_utils import convert_df_to_training_format
import scipy.cluster.hierarchy

from allrank.models.losses.neuralNDCG import neuralNDCG, neuralNDCG_transposed


class NeuralNDCGLoss:
    """
    Simple class wrapper of neuralNDCG loss
    """

    def __init__(self, **loss_kwargs):
        self.loss_kwargs = loss_kwargs

    def __call__(self, pred, gt):
        return neuralNDCG(pred, gt, **self.loss_kwargs)


def run_torch_model_training(X, Y, stratification_classes,
                             max_epochs=1000,
                             batch_size=32,
                             learning_rate=0.001,
                             num_cross_val_folds=10,
                             num_hidden_layer_features=32
                             ):
    # Initial definitions:
    train_val_dataset = TrainValData(X.astype("float32"), Y.astype("float32"))
    test_data = TestData(X.astype("float32"))
    all_results = pd.DataFrame()
    NDCG_rank_loss = NeuralNDCGLoss()
    # soresen_loss = F.binary_cross_entropy_with_logits

    # Define cross-val split:
    skf = sklearn.model_selection.StratifiedKFold(n_splits=num_cross_val_folds)
    skf.get_n_splits()
    pbar_cross_split = tqdm.tqdm(skf.split(range(X.shape[0]), stratification_classes),
                             leave=False, total=num_cross_val_folds)

    # Loop over cross-val folds:
    for fold, (train_index, test_index) in enumerate(pbar_cross_split):
        # Define data-loaders:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_index.tolist())
        valid_sampler = torch.utils.data.SubsetRandomSampler(test_index.tolist())
        # TODO: increase number of workers?
        num_workers = 0
        train_loader = DataLoader(dataset=train_val_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_workers)
        val_loader = DataLoader(dataset=train_val_dataset, batch_size=batch_size, sampler=valid_sampler,
                                num_workers=num_workers)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, sampler=valid_sampler,
                                 num_workers=num_workers)

        # Define model:
        model = SimpleTwoLayersNN(num_feat=num_hidden_layer_features,
                                  nb_in_feat=X.shape[1],
                                  nb_out_feat=Y.shape[1],
                                  loss=NDCG_rank_loss,
                                  learning_rate=learning_rate)

        # Train model using Lightning:
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        trainer = pl.Trainer(
            callbacks=[lr_monitor],
            default_root_dir=str(Path.cwd() / "training_data_torch"),
            gradient_clip_algorithm="norm",
            enable_progress_bar=True,
            max_epochs=max_epochs,
            detect_anomaly=True,
            log_every_n_steps=5,
            #    auto_lr_find=True,
            # ckpt_path="path",
        )
        trainer.fit(model=model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)

        # Predict and save predictions on test set:
        prediction = trainer.predict(model, dataloaders=test_loader)
        nb_batches = len(prediction)
        pred_array = np.concatenate([tensor[0].numpy() for tensor in prediction])
        pred_indices = np.concatenate([tensor[1].numpy() for tensor in prediction])

        lc_results = pd.DataFrame(pred_array, index=pred_indices)
        lc_results["fold"] = fold
        all_results = pd.concat([all_results, lc_results])

    return all_results


def train_NN_with_rank_loss(intensities_df,
                            features_df,
                            adducts_one_hot,
                            do_feature_selection=False,
                            path_feature_importance_csv=None,
                            num_cross_val_folds=10,
                            intensity_column="norm_intensity"
                            ):
    assert not do_feature_selection, "Feature selection not implemented yet"
    assert path_feature_importance_csv is None, "Feature selection not implemented yet"

    # -------------------------
    # Reshape data to training format:
    # -------------------------
    Y = intensities_df.pivot(index=['name_short', 'adduct'], columns=["matrix", "polarity"], values=intensity_column)
    Y_detected = intensities_df.pivot(index=['name_short', 'adduct'], columns=["matrix", "polarity"], values="detected")

    # Mask NaN values (ion-intensity values not provided for a given matrix-polarity):
    Y_is_na = Y_detected.isna()
    Y_detected[Y_is_na] = False

    # Remove ions that are never detected:
    detected_ion_mask = Y_detected.sum(axis=1) > 0
    Y = Y[detected_ion_mask]
    Y_detected = Y_detected[detected_ion_mask]
    Y_is_na = Y_is_na[detected_ion_mask]

    # Set not-detected intensities to zero:
    Y[Y_detected == False] = 0

    # Get feature array used for training:
    X = pd.DataFrame(features_df.loc[Y.index.get_level_values(0)].to_numpy(), index=Y.index)
    X = X.join(pd.DataFrame(adducts_one_hot.loc[Y.index.get_level_values(1)].to_numpy(), index=Y.index),
               how="inner", rsuffix="adduct")

    # -------------------------
    # Find stratification classes and start training:
    # -------------------------
    # TODO: use alternative stratification
    Z_clust = scipy.cluster.hierarchy.linkage(Y, method="ward")
    out_clustering = scipy.cluster.hierarchy.fcluster(Z_clust, t=9, criterion="distance")

    # # Sorting should not be necessary...
    # masked_Y = np.ma.masked_array(Y.to_numpy(), mask=Y_detected.to_numpy())
    # masked_Y.argsort(axis=1, fill_value=-1)
    # np.ma.argsort()
    # np.argsort(Y.to_numpy(), axis=1, )

    # Not-detected intensities are masked to value -1 and will be ignored in the ranking loss:
    Y[Y_is_na] = -1

    out = run_torch_model_training(X.to_numpy(), Y.to_numpy(),
                                       stratification_classes=out_clustering,
                                       num_cross_val_folds=num_cross_val_folds,
                                       max_epochs=20,
                                       # max_epochs=1,
                                       batch_size=8,
                                       # batch_size=32,
                                       learning_rate=0.01,
                                       num_hidden_layer_features=32)


    # TODO: reshape results
    # Reshape results:
    matrix_multi_index = Y.columns
    training_results = out.sort_index()
    # Set index with molecule/adduct names:
    training_results = pd.DataFrame(training_results.to_numpy(), index=Y.index, columns=training_results.columns)
    reshaped_gt = Y.stack([i for i in range(len(matrix_multi_index.levels))], )
    reshaped_gt.name = "observed_value"
    reshaped_prediction = pd.DataFrame(training_results.drop(columns="fold").to_numpy(), index=Y.index, columns=matrix_multi_index).stack([i for i in range(len(matrix_multi_index.levels))])
    reshaped_prediction.name = "prediction"
    reshaped_out = reshaped_gt.to_frame().join(reshaped_prediction.to_frame(), how="inner")

    # Add back fold info:
    reshaped_out["fold"] = training_results.loc[[i for i in zip(reshaped_out.index.get_level_values(0), reshaped_out.index.get_level_values(1))], "fold"].to_numpy()
    reshaped_out["model_type"] = "NN"
    reshaped_out.reset_index(inplace=True)

    return reshaped_out
