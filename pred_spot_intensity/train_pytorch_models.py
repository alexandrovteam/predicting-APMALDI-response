import os
from pathlib import Path

import pandas as pd
import numpy as np
import sklearn
import tqdm
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import logging
from sklearn.neural_network import MLPClassifier, MLPRegressor

import shap

try:
    import torch.nn.functional as F
    import pytorch_lightning as pl
    import torch.nn as nn
    import torch
    from pytorch_lightning.callbacks import LearningRateMonitor
    from torch.utils.data import DataLoader
    from pred_spot_intensity.pytorch_utils import TrainValData, TestData, SorensenDiceLoss, SimpleTwoLayersNN, RankingLossWrapper
    import skorch
except ImportError:
    torch = None
    skorch = None
import scipy.cluster.hierarchy

try:
    # from allrank.models.losses.neuralNDCG import neuralNDCG, neuralNDCG_transposed
    # from allrank.models.losses.lambdaLoss import lambdaLoss as rankingLossFct
    from allrank.models.losses.rankNet import rankNet as rankingLossFct
except ImportError:
    neuralNDCG = None
    rankingLossFct = None


# TODO: give as argument
DEVICE = "cpu"




def train_torch_model(model, train_loader, val_loader=None, test_loader=None,
                      max_epochs=1000, verbose=True):
    # Train model using Lightning:
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        callbacks=[lr_monitor] if verbose else None,
        default_root_dir=str(Path.cwd() / "training_data_torch"),
        gradient_clip_algorithm="norm",
        enable_progress_bar=True if verbose else False,
        enable_model_summary=True if verbose else False,
        enable_checkpointing=True if verbose else False,
        logger=True if verbose else False,  # Disable tensorboard logs
        max_epochs=max_epochs,
        detect_anomaly=True,
        log_every_n_steps=5,
        accelerator=DEVICE,
        # gpus=0,
        #    auto_lr_find=True,
        # ckpt_path="path",
    )
    if not verbose:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    if test_loader is not None:
        # Predict and save predictions on test set:
        prediction = trainer.predict(model, dataloaders=test_loader)
        # nb_batches = len(prediction)
        pred_array = np.concatenate([tensor[0].numpy() for tensor in prediction])
        pred_indices = np.concatenate([tensor[1].numpy() for tensor in prediction])
        return trainer, pred_array, pred_indices
    else:
        return trainer


def train_pytorch_model_wrapper(train_x, test_x=None, train_y=None, test_y=None,
                                type_of_models="regressor",
                                out_multi_index=None,
                                model_set=(),
                                name_test=None,
                                train=True,
                                test_multioutout_models=True,
                                y_is_multioutput=True,
                                max_epochs=50,
                                batch_size=8,
                                learning_rate=0.001,
                                num_hidden_layer_features=32,
                                do_feature_selection=False,
                                feature_names=None,
                                matrix=None,
                                polarity=None,
                                molecule_names=None,
                                feature_selection_out_dir=None
                                ):
    """
    Temp wrapper for compatibility with sklearn model training
    """
    assert out_multi_index is None
    assert model_set == ()
    assert train
    assert not y_is_multioutput

    do_test = test_x is not None

    if train_y.ndim == 1:
        train_y = train_y[:, None]
        if do_test: test_y = test_y[:, None]

    # Initial definitions:
    # ignore_mask = ignore_mask if ignore_mask is None else ignore_mask.astype("float32")
    if type_of_models == "classifier":
        # TODO: make sure that this is up to date
        # final_activation = nn.Sigmoid()
        # loss_function = SorensenDiceLoss()
        # loss_function = nn.BCELoss()
        # soresen_loss = F.binary_cross_entropy_with_logits
        # skorch_trainer_class = skorch.NeuralNetClassifier
        sklearn_model = MLPClassifier(max_iter=2000)
    elif type_of_models == "regressor":
        # final_activation = None
        # loss_function = nn.MSELoss()
        # skorch_trainer_class = skorch.NeuralNetRegressor
        # loss_function = nn.L1Loss() # Does not work
        sklearn_model = MLPRegressor(max_iter=2000)
    else:
        raise ValueError(type_of_models)

    # ----------------------------
    # sklearn:
    # ----------------------------
    if type_of_models == "classifier":
        train_y = train_y.astype("int64")[:, 0]
    sklearn_model.fit(train_x, train_y)
    model = sklearn_model

    if do_feature_selection:
        assert feature_selection_out_dir is not None
        if not isinstance(feature_selection_out_dir, Path): feature_selection_out_dir = Path(feature_selection_out_dir)
        feature_selection_out_dir = feature_selection_out_dir / type_of_models
        feature_selection_out_dir.mkdir(exist_ok=True, parents=True)

        # --------------
        # SHAP:
        # --------------
        train_x_summary = shap.kmeans(train_x, 10)
        explainer = shap.KernelExplainer(
            model.predict,
            train_x_summary
            # train_x[np.random.choice(np.arange(train_x.shape[0]), 100, replace=False)]
        )
        shap_values = explainer.shap_values(train_x)
        # OLD PYTORCH code:
        # explainer = shap.DeepExplainer(
        #     model,
        #     torch.from_numpy(train_x[np.random.choice(np.arange(train_x.shape[0]), 100, replace=False)]
        #                      ).to(DEVICE).float())
        # shap_values = explainer.shap_values(
        #     torch.from_numpy(train_x).to(DEVICE).float()
        # )

        # Filter out the features related to adducts:
        features_mask = ["adduct" not in feat for feat in feature_names]
        selected_features = [feat for feat in feature_names if "adduct" not in feat]

        # explainer = shap.KernelExplainer(model.predict, X_train)
        # shap_values = explainer.shap_values(X_test, nsamples=100)
        shap.initjs()
        # for i in range(10):
        #     fig = plt.gcf()
        #     shap.force_plot(explainer.expected_value, shap_values[i, features_mask], np.around(train_x[i, features_mask], decimals=2), feature_names=selected_features, matplotlib=True, show=False)
        #     plt.savefig(f'/Users/alberto-mac/EMBL_repos/spotting-project-regression/plots/feature_importance/{matrix}_{polarity}_{i}_force_plot.png')

        # plot the explanation of the first prediction
        # Note the model is "multi-output" because it is rank-2 but only has one column
        # shap.force_plot(explainer.expected_value[0], shap_values[0][0], x_test_words[0])
        shap.summary_plot(shap_values[:, features_mask], train_x[:, features_mask], feature_names=selected_features, show=False)
        plt.savefig(feature_selection_out_dir / f'{matrix}_{polarity}_summary_plot.png')

        # Make sure that matplolib clears the history for next plots:
        plt.show(block=False)
        fig = plt.gcf()
        fig.gca().clear()
        plt.cla()
        plt.close(fig)


        # Collect stats for high and low features separately:
        filtered_x = train_x[:, features_mask]
        # Normalize features between 0 and 1:
        norm_x = filtered_x - filtered_x.min(axis=0)
        norm_x /= norm_x.max(axis=0)
        high_value_x = norm_x > 0.5

        # Now return the actual values
        result_df = pd.DataFrame({
            "Feature importance (mean abs value of shap-value)": np.mean(np.abs(shap_values[:, features_mask]), axis=0),
            "Model-impact of points with high-value features": np.mean(shap_values[:, features_mask][high_value_x], axis=0),
            "Model-impact of points with low-value features": np.mean(shap_values[:, features_mask][~high_value_x], axis=0),
            # "std_abs_shap_value": np.std(np.abs(shap_values[:, features_mask]), axis=0),
            # "matrix": matrix,
            # "polarity": polarity,
            "Feature name": selected_features,
            # "adduct": "all",
        })

        # if "adduct" in molecule_names.columns.to_list():
        #     molecule_names.reset_index(drop=True, inplace=True)
        #     for adduct in molecule_names.adduct.unique():
        #         molecule_mask = (molecule_names.adduct == adduct).to_list()
        #         plt.show()
        #         fig = plt.gcf()
        #         shap.summary_plot(shap_values[molecule_mask][:, features_mask], train_x[molecule_mask][:, features_mask], feature_names=selected_features,
        #                           show=False)
        #         plt.savefig(out_plot_dir / f'summary_plot_{adduct}.png')
        #
        #         loc_df = pd.DataFrame({
        #             "Feature importance (avg abs value of shap-value)": np.mean(np.abs(shap_values[molecule_mask][:, features_mask]), axis=0),
        #             # "std_abs_shap_value": np.std(np.abs(shap_values[molecule_mask][:, features_mask]), axis=0),
        #             # "matrix": matrix,
        #             # "polarity": polarity,
        #             "Feature name": selected_features,
        #             "adduct": adduct,
        #         })
        #         result_df = pd.concat([result_df, loc_df])

        return result_df

    # # Define data-loaders:
    # train_dataset = TrainValData(train_x.astype("float32"), train_y.astype("float32"))
    # val_dataset = TrainValData(test_x.astype("float32"), test_y.astype("float32"))
    # test_dataset = TestData(test_x.astype("float32"))
    # num_workers = 0
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
    #                           num_workers=num_workers, drop_last=True)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
    #                           num_workers=num_workers)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
    #                           num_workers=num_workers)

    # _, pred_array, pred_indices = train_torch_model(model, train_loader=train_loader,
    #                                                 val_loader=val_loader,
    #                                                 test_loader=test_loader,
    #                                                 max_epochs=max_epochs, verbose=False)
    #
    # # Sort predictions in original order:
    # predictions = pred_array[:, 0][pred_indices]

    # if do_test:
    #     predictions = skorch_trainer.predict_proba(test_x.astype("float32"))
    #     results_df = pd.DataFrame({"prediction": predictions[:, 0] if predictions.ndim > 1 else predictions,
    #                            "observed_value": test_y[:, 0] if test_y.ndim > 1 else test_y,
    #                            type_of_models: "pytorch_NN"})
    #
    #     if name_test is not None:
    #         results_df = results_df.merge(name_test, left_index=True, right_index=True)
    #
    #     return results_df


def train_torch_model_cross_val_loop(X, Y, task_name,
                                     stratification_classes,
                                     ignore_mask=None,
                                     max_epochs=1000,
                                     batch_size=32,
                                     learning_rate=0.001,
                                     num_cross_val_folds=10,
                                     num_hidden_layer_features=32
                                     ):
    """
    TODO: Refactor arguments: remove task name and add final_activation/loss as arguments
    """
    # Initial definitions:
    ignore_mask = ignore_mask if ignore_mask is None else ignore_mask.astype("float32")
    train_val_dataset = TrainValData(X.astype("float32"), Y.astype("float32"),
                                     ignore_mask=ignore_mask)
    test_data = TestData(X.astype("float32"))
    all_results = pd.DataFrame()
    skorch_trainer_class = skorch.NeuralNetRegressor
    if task_name == "ranking":
        final_activation = None
        loss_function = RankingLossWrapper()
    elif task_name == "detection":
        final_activation = nn.Sigmoid()
        loss_function = SorensenDiceLoss()
        # soresen_loss = F.binary_cross_entropy_with_logits
    else:
        raise ValueError(task_name)

    # Define cross-val split:
    skf = sklearn.model_selection.StratifiedKFold(n_splits=num_cross_val_folds)
    skf.get_n_splits()
    pbar_cross_split = tqdm.tqdm(skf.split(range(X.shape[0]), stratification_classes),
                                 leave=False, total=num_cross_val_folds)

    # Loop over cross-val folds:
    for fold, (train_index, test_index) in enumerate(pbar_cross_split):
        train_x = X[train_index]
        train_y = Y[train_index]
        # Define model:
        model = SimpleTwoLayersNN(num_feat=num_hidden_layer_features,
                                  nb_in_feat=X.shape[1],
                                  nb_out_feat=Y.shape[1],
                                  final_activation=final_activation,
                                  # keep_channel_dim_out=type_of_models != "classifier"
                                  )

        skorch_trainer = skorch_trainer_class(
            model,
            max_epochs=300,
            lr=0.1, # TODO: update
            train_split=None,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            criterion=loss_function,
            verbose=True)

        # if type_of_models == "classifier":
        #     train_y = train_y.astype("int64")[:, 0]
        # train_y = train_y.astype("float32")[:, 0]
        # else:
        train_y = train_y.astype("float32")
        skorch_trainer.fit(train_x.astype("float32"), train_y)

        # Prediction on test:
        test_x = X[test_index]
        # test_y = Y[test_index]
        predictions = skorch_trainer.predict_proba(test_x.astype("float32"))
        lc_results = pd.DataFrame(predictions, index=test_index)

        # -------------------------------
        # OLD PYTORCH LIGHTNING TRAINING:
        # # Define data-loaders:
        # train_sampler = torch.utils.data.SubsetRandomSampler(train_index.tolist())
        # valid_sampler = torch.utils.data.SubsetRandomSampler(test_index.tolist())
        # num_workers = 0
        # train_loader = DataLoader(dataset=train_val_dataset, batch_size=batch_size, sampler=train_sampler,
        #                           num_workers=num_workers, drop_last=True)
        # val_loader = DataLoader(dataset=train_val_dataset, batch_size=batch_size, sampler=valid_sampler,
        #                         num_workers=num_workers)
        # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, sampler=valid_sampler,
        #                          num_workers=num_workers)
        #
        # # Define model:
        # model = SimpleTwoLayersNN(num_feat=num_hidden_layer_features,
        #                           nb_in_feat=X.shape[1],
        #                           nb_out_feat=Y.shape[1],
        #                           loss=loss_function,
        #                           learning_rate=learning_rate,
        #                           final_activation=final_activation,
        #                           has_ignore_mask=ignore_mask is not None)
        #
        # _, pred_array, pred_indices = train_torch_model(model, train_loader, val_loader, test_loader, max_epochs)
        # lc_results = pd.DataFrame(pred_array, index=pred_indices)
        # -------------------------------

        lc_results["fold"] = fold
        all_results = pd.concat([all_results, lc_results])

    return all_results


def features_selection_torch_model(X, Y, task_name,
                                   ignore_mask=None,
                                   max_epochs=1000,
                                   batch_size=32,
                                   learning_rate=0.001,
                                   num_cross_val_folds=10,
                                   num_hidden_layer_features=32,
                                   checkpoint_path=None,
                                   only_train=False
                                   ):
    """
    TODO: Refactor arguments: remove task name and add final_activation/loss as arguments
    """
    # Initial definitions:
    ignore_mask = ignore_mask if ignore_mask is None else ignore_mask.astype("float32")
    train_val_dataset = TrainValData(X.astype("float32"), Y.astype("float32"),
                                     ignore_mask=ignore_mask)
    test_data = TestData(X.astype("float32"))
    all_results = pd.DataFrame()
    if task_name == "ranking":
        final_activation = None
        loss_function = RankingLossWrapper()
    elif task_name == "detection":
        final_activation = nn.Sigmoid()
        loss_function = SorensenDiceLoss()
        # soresen_loss = F.binary_cross_entropy_with_logits
    elif task_name == "regression":
        final_activation = None
        loss_function = nn.MSELoss()
        # loss_function = nn.L1Loss() # Does not work
    else:
        raise ValueError(task_name)

    # Define data-loaders:
    num_workers = 0
    train_loader = DataLoader(dataset=train_val_dataset, batch_size=batch_size,
                              num_workers=num_workers, drop_last=True)

    # Define model:
    if checkpoint_path is None:
        model = SimpleTwoLayersNN(num_feat=num_hidden_layer_features,
                                  nb_in_feat=X.shape[1],
                                  nb_out_feat=Y.shape[1],
                                  loss=loss_function,
                                  learning_rate=learning_rate,
                                  final_activation=final_activation,
                                  has_ignore_mask=ignore_mask is not None)

        trainer = train_torch_model(model, train_loader, max_epochs=max_epochs)  # TODO: epochs
        # model = trainer.model
        from torchmetrics import SpearmanCorrCoef
        spearman = SpearmanCorrCoef()

        test_loader = DataLoader(dataset=test_data, batch_size=batch_size,
                                 num_workers=num_workers)
        predictions = trainer.predict(model, dataloaders=test_loader)
        ignore_mask_flatten = ignore_mask.flatten().astype("bool")
        pred_array = np.concatenate([tensor[0].numpy() for tensor in predictions]).astype("float32").flatten()[
            ~ignore_mask_flatten]
        print("Score MSE: ", scipy.stats.spearmanr(pred_array, Y.astype("float32").flatten()[~ignore_mask_flatten]))
    else:
        model = SimpleTwoLayersNN.load_from_checkpoint(checkpoint_path)

    e = shap.DeepExplainer(
        model,
        torch.from_numpy(X[np.random.choice(np.arange(X.shape[0]), 100, replace=False)]
                         ).to(DEVICE).float())
    shap_values = e.shap_values(
        torch.from_numpy(X).to(DEVICE).float()
    )

    return shap_values


def train_pytorch_model_on_intensities(intensities_df,
                                       features_df,
                                       adducts_one_hot,
                                       task_name,
                                       do_feature_selection=False,
                                       path_feature_importance_csv=None,
                                       num_cross_val_folds=10,
                                       intensity_column="norm_intensity",
                                       checkpoint_path=None,
                                       use_adduct_features=True,
                                       adducts_columns=None
                                       ):
    raise DeprecationWarning
    assert torch is not None

    # -------------------------
    # Reshape data to training format:
    # -------------------------
    index_cols = ['name_short', 'adduct'] if use_adduct_features else ['name_short']
    Y = intensities_df.pivot(index=index_cols, columns=["matrix", "polarity"], values=intensity_column)
    Y_detected = intensities_df.pivot(index=index_cols, columns=["matrix", "polarity"], values="detected")

    # Mask NaN values (ion-intensity32 values not provided for a given matrix-polarity):
    Y_is_na = Y_detected.isna()
    Y_detected[Y_is_na] = False

    detected_ion_mask = Y_detected.sum(axis=1) > 0
    if task_name == "ranking":
        # Remove ions that are never detected:
        Y = Y[detected_ion_mask]
        Y_detected = Y_detected[detected_ion_mask]
        Y_is_na = Y_is_na[detected_ion_mask]

    # Set not-detected intensities to zero:
    Y[Y_detected == False] = 0

    # Get feature array used for training:
    X = pd.DataFrame(features_df.loc[Y.index.get_level_values(0)].to_numpy(), index=Y.index)
    if use_adduct_features:
        X = X.join(pd.DataFrame(adducts_one_hot.loc[Y.index.get_level_values(1)].to_numpy(), index=Y.index),
                   how="inner", rsuffix="adduct")

    # -------------------------
    # Find stratification classes and start training:
    # -------------------------
    # Z_clust = scipy.cluster.hierarchy.linkage(Y, method="ward")
    # out_clustering = scipy.cluster.hierarchy.fcluster(Z_clust, t=9, criterion="distance")

    kmeans = KMeans(n_clusters=10, random_state=45).fit(Y)
    out_clustering = kmeans.labels_

    # # Sorting should not be necessary...
    # masked_Y = np.ma.masked_array(Y.to_numpy(), mask=Y_detected.to_numpy())
    # masked_Y.argsort(axis=1, fill_value=-1)
    # np.ma.argsort()
    # np.argsort(Y.to_numpy(), axis=1, )

    ignore_mask = None
    if task_name == "ranking":
        # Not-detected intensities are masked to value -1 and will be ignored in the ranking loss:
        Y[Y_is_na] = -1
    elif task_name == "detection":
        Y = Y_detected
        ignore_mask = Y_is_na.to_numpy()
    elif task_name == "regression":
        # Mask not-detected ones:
        ignore_mask = np.logical_not(Y_detected.to_numpy())
    else:
        raise ValueError

    # -------------------------
    # Train:
    # -------------------------
    if not do_feature_selection:
        out = train_torch_model_cross_val_loop(X.to_numpy(), Y.to_numpy(),
                                               task_name,
                                               stratification_classes=out_clustering,
                                               ignore_mask=ignore_mask,
                                               num_cross_val_folds=num_cross_val_folds,
                                               max_epochs=20,
                                               # max_epochs=1,
                                               batch_size=8,
                                               # batch_size=32,
                                               learning_rate=0.01,
                                               num_hidden_layer_features=32)
        # -------------------------
        # Reshape results:
        # -------------------------
        matrix_multi_index = Y.columns
        training_results = out.sort_index()
        # Set index with molecule/adduct names:
        training_results = pd.DataFrame(training_results.to_numpy(), index=Y.index, columns=training_results.columns)
        reshaped_gt = Y.stack([i for i in range(len(matrix_multi_index.levels))], )
        reshaped_gt.name = "observed_value"
        reshaped_prediction = pd.DataFrame(training_results.drop(columns="fold").to_numpy(), index=Y.index,
                                           columns=matrix_multi_index).stack(
            [i for i in range(len(matrix_multi_index.levels))])
        reshaped_prediction.name = "prediction"
        reshaped_out = reshaped_gt.to_frame().join(reshaped_prediction.to_frame(), how="inner")

        # Add back fold info:
        reshaped_out["fold"] = training_results.loc[[i for i in zip(reshaped_out.index.get_level_values(0),
                                                                    reshaped_out.index.get_level_values(
                                                                        1))], "fold"].to_numpy()
        reshaped_out["model_type"] = "NN"
        reshaped_out.reset_index(inplace=True)

        return reshaped_out

    else:
        shap_values = features_selection_torch_model(X.to_numpy(), Y.to_numpy(),
                                                     task_name,
                                                     ignore_mask=ignore_mask,
                                                     num_cross_val_folds=num_cross_val_folds,
                                                     max_epochs=100,  # TODO: change
                                                     # max_epochs=1,
                                                     batch_size=8,
                                                     # batch_size=32,
                                                     learning_rate=0.01,
                                                     num_hidden_layer_features=128,
                                                     checkpoint_path=checkpoint_path)
        print("done")

        feat_names = features_df.columns.tolist()
        if use_adduct_features:
            feat_names = adducts_columns.tolist() + feat_names

        matrix_multi_index = Y.columns
        ignore_mask = ignore_mask.astype("bool")
        # Filter out ignored items:
        # filtered_shap_values = [shap_values[matr_idx][ignore_mask[:,matr_idx]] for matr_idx in range(matrix_multi_index.shape[0])]
        # TODO: ignore stuff in the mask
        # df = pd.DataFrame(columns=["mean_abs_shap", "matrix", "polarity", "feat_name"])
        df_feat_importance = pd.DataFrame()
        for i, (matrix, polarity) in enumerate(matrix_multi_index):
            loc_shap_values = shap_values[i][~ignore_mask[:, i]]
            loc_df = pd.DataFrame({
                "mean_abs_shap": np.mean(np.abs(loc_shap_values), axis=0),
                "matrix": matrix,
                "polarity": polarity,
                "feature_name": feat_names
            })
            df_feat_importance = pd.concat([df_feat_importance, loc_df])
        df_feat_importance = df_feat_importance.sort_values("mean_abs_shap", ascending=False).set_index("feature_name",
                                                                                                        drop=True)
        return df_feat_importance
