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

from pred_spot_intensity.pytorch_utils import TrainValData, TestData, SorensenDiceLoss, BinaryClassification
from pred_spot_intensity.sklearn_training_utils import convert_df_to_training_format
import scipy.cluster.hierarchy


def run_torch_model_training(X, Y, stratification_classes):
    train_val_dataset = TrainValData(X.astype("float32"), Y.astype("float32"))
    test_data = TestData(X.astype("float32"))

    # EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    soresen_loss = SorensenDiceLoss()
    # soresen_loss = F.binary_cross_entropy_with_logits

    NUM_FEAT = 32

    # Do stratification and cross-val-loop:

    NUM_SPLITS = 10

    # TODO: if SPOLITS == 2, use other function...?
    skf = sklearn.model_selection.StratifiedKFold(n_splits=NUM_SPLITS)
    skf.get_n_splits()

    pbar_cross_split = tqdm.tqdm(skf.split(range(X.shape[0]), stratification_classes),
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
    Y_detected[Y_detected.isna()] = False

    # Remove ions that are never detected:
    detected_ion_mask = Y_detected.sum(axis=1) > 0
    Y = Y[detected_ion_mask]
    Y_detected = Y_detected[detected_ion_mask]

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

    # # Sorting should not be stricktly necessary...
    # masked_Y = np.ma.masked_array(Y.to_numpy(), mask=Y_detected.to_numpy())
    # masked_Y.argsort(axis=1, fill_value=-1)
    # np.ma.argsort()
    # np.argsort(Y.to_numpy(), axis=1, )

    # Not-detected intensities are masked to value -1 and will be ignored in the ranking loss:
    Y[Y_detected == False] = -1

    results = run_torch_model_training(X, Y,
                                       stratification_classes=out_clustering)

# def train_multi_output_regressors(
#         intensity_df,
#         digitized_mol_properties,
#         features_normalized,
#         adducts_columns=None,
#         intensity_column="spot_intensity",
#         test_split_col_name='mol_strat_class',
#         num_cross_val_folds=10
# ):
#     # raise DeprecationWarning()
#     # Cross-validation loop:
#     regression_results = pd.DataFrame()
#     selected_mols = digitized_mol_properties
#
#     sorted_intensities = intensity_df.sort_values(by=['name_short', 'adduct', "matrix", "polarity"])
#     sorted_intensities = pd.merge(sorted_intensities,
#                                   features_normalized,
#                                   how="left",
#                                   right_index=True,
#                                   left_on="name_short"
#                                   )
#
#     # Create folds, making sure that there are enough detected observations for each matrix:
#     skf = sklearn.model_selection.StratifiedKFold(n_splits=num_cross_val_folds)
#     pbar_cross_split = tqdm.tqdm(skf.split(selected_mols.index, selected_mols[test_split_col_name]), leave=False,
#                                  total=num_cross_val_folds)
#
#     for fold, (train_index, test_index) in enumerate(pbar_cross_split):
#         train_intensities = sorted_intensities[sorted_intensities.name_short.isin(selected_mols.index[train_index])]
#         test_intensities = sorted_intensities[sorted_intensities.name_short.isin(selected_mols.index[test_index])]
#         g_train = train_intensities.groupby(by=['name_short', 'adduct'])
#         g_test = test_intensities.groupby(by=['name_short', 'adduct'])
#         train_y = np.array([rows.to_numpy() for _, rows in g_train[intensity_column]])
#         train_x = np.array([rows.iloc[0].to_numpy() for _, rows in
#                             g_train[adducts_columns.tolist() + features_normalized.columns.tolist()]])
#
#         test_y = np.array([rows.to_numpy() for _, rows in g_test[intensity_column]])
#         test_x = np.array([rows.iloc[0].to_numpy() for _, rows in
#                            g_test[adducts_columns.tolist() + features_normalized.columns.tolist()]])
#
#         test_mol_names = pd.DataFrame([[name, adduct] for (name, adduct), _ in g_test],
#                                       columns=["name_short", "adduct"])
#
#         matrix_names = train_intensities[["matrix", "polarity"]].drop_duplicates()
#         out_multi_index = pd.MultiIndex.from_arrays([matrix_names["matrix"], matrix_names["polarity"]])
#         results_df = train_multiple_models(train_x, test_x, train_y, test_y,
#                                            type_of_models="regressor",
#                                            name_test=test_mol_names,
#                                            out_multi_index=out_multi_index,
#                                            train=True)
#         results_df["fold"] = int(fold)
#         regression_results = pd.concat([regression_results, results_df])
#     return regression_results

# def train_multiple_models(train_x, test_x, train_y, test_y,
#                           type_of_models="regressor",
#                           out_multi_index=None,
#                           model_set=(),
#                           name_test=None,
#                           train=True,
#                           test_multioutout_models=True,
#                           y_is_multioutput=True
#                           ):
#     if y_is_multioutput: assert out_multi_index is not None
#
#     results_df = pd.DataFrame()
#     all_models = sets_of_models[type_of_models]
#
#     if len(model_set) == 0: model_set = all_models.keys()
#
#     if name_test is not None and y_is_multioutput:
#         nb_outputs = test_y.shape[1]
#         name_test = name_test.loc[name_test.index.repeat(nb_outputs)]
#
#     pbar = tqdm(model_set, leave=False)
#     # for r in model_set:
#     for r in pbar:
#         pbar.set_postfix({type_of_models: r})
#         regressor = all_models[r]
#         # These classifiers need to train several models (one per class):
#         if "MultiOut" not in r and y_is_multioutput:
#             regressor = MultiOutputRegressor(regressor)
#         elif "MultiOut" in r and (not test_multioutout_models or not y_is_multioutput):
#             # Skip MultiOutput models:
#             continue
#         if train:
#             regressor.fit(train_x, train_y)
#             y_pred = regressor.predict(test_x)
#
#             if y_is_multioutput:
#                 loc_res_df = pd.DataFrame(test_y, columns=out_multi_index).stack(
#                     [i for i in range(len(out_multi_index.levels))]).reset_index().drop(columns=["level_0"]).rename(
#                     columns={0: "observed_value"})
#                 loc_res_df['prediction'] = pd.DataFrame(y_pred, columns=out_multi_index).stack(
#                     [i for i in range(len(out_multi_index.levels))]).reset_index()[0]
#                 loc_res_df[type_of_models] = r
#                 loc_res_df.reset_index(drop=True, inplace=True)
#                 if name_test is not None:
#                     loc_res_df = loc_res_df.merge(name_test.reset_index(drop=True), left_index=True, right_index=True)
#             else:
#                 loc_res_df = pd.DataFrame({'observed_value': test_y,
#                                            'prediction': y_pred,
#                                            type_of_models: r})
#                 if name_test is not None:
#                     loc_res_df = loc_res_df.merge(name_test, left_index=True, right_index=True)
#         else:
#             raise NotImplementedError()
#         results_df = pd.concat([results_df, loc_res_df])
#
#     results_df = results_df.reset_index(drop=True)
#     return results_df
