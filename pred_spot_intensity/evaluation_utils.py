import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import scipy.stats
from sklearn.metrics import mean_squared_error

import matplotlib


def get_scores(data, task_type="classification"):
    if task_type == "classification":
        clf_report = classification_report(data.observed_value, data.prediction, output_dict=True, zero_division=0)
        scores = [clf_report['0']['precision'], clf_report['0']['recall'],
                  clf_report['0']['f1-score'], clf_report['0']['support'],
                  clf_report['1']['precision'], clf_report['1']['recall'],
                  clf_report['1']['f1-score'], clf_report['1']['support'],
                  clf_report['macro avg']['f1-score']]
        if data.observed_value.max() == 0:
            print("no detection!")
    elif task_type == "regression":
        spearman = scipy.stats.spearmanr(data.observed_value, data.prediction)
        pearson = scipy.stats.pearsonr(data.observed_value, data.prediction)
        mse = mean_squared_error(data.observed_value, data.prediction, squared=False)
        mse_std = mse / data['observed_value'].std()
        scores = [spearman[0], spearman[1],
                  pearson[0], pearson[1], mse, mse_std,
                  data[data['observed_value'] != 0].shape[0]]
    else:
        raise ValueError(task_type)
    return scores


def get_molecules_with_zero_formal_charge(raw_intensities_csv):
    raw_intensities_df = pd.read_csv(raw_intensities_csv, index_col=0)
    return raw_intensities_df[raw_intensities_df.formal_charge == 0.0]["name_short"].drop_duplicates().to_list()


def compute_scores(filenames, task_name, task_type,
                   eval_feat_selection=False,
                   test_all_adduct_setups=False,
                   load_multiple_thresholds=False,
                   thresh_column_name=None,
                   has_multiple_iterations=False,
                   name_global_column="global",
                   molecules_with_zero_charge=None,
                   results_dir=None):
    assert not (eval_feat_selection and load_multiple_thresholds)
    assert not (eval_feat_selection and has_multiple_iterations)
    assert not (load_multiple_thresholds and has_multiple_iterations)
    if load_multiple_thresholds:
        assert thresh_column_name is not None
    main_col_names = ['train_setup', 'model_type', 'adduct_setup']
    if eval_feat_selection: main_col_names += ['feat_sel_method', 'quantile', 'nb_features']
    if load_multiple_thresholds: main_col_names += [thresh_column_name]
    if has_multiple_iterations: main_col_names += ['iter_index']

    adduct_setups_settings = {
        "All adducts": {
            "adducts": ['-H', '+Cl', '[M]-', '+H', '[M]+', '+K', '+Na'],
            "only_zero_formal_charge": False
        }}
    if test_all_adduct_setups:
        extra_setups = {
            "Only '-H'/'+H' adducts and metabolites with zero formal charge": {
                "adducts": ['-H', '+H'],
                "only_zero_formal_charge": True
            },
            "No radicals adducts '[M]+'/'[M]-' and only metabolites with zero formal charge": {
                "adducts": ['-H', '+Cl', '+H', '+K', '+Na'],
                "only_zero_formal_charge": True
            }
        }
        adduct_setups_settings.update(extra_setups)

    all_model_typenames = np.array([
        "classifier",
        "regressor",
        "model_type"
    ])

    if "detection" in task_name:
        score_cols = ['not_det_precision', 'not_det_recall', 'not_det_f1', 'not_detected',
                      'det_precision', 'det_recall', 'det_f1', 'detected', 'macro_avg_f1_score']
    elif "regression" in task_name or "rank" in task_name:
        score_cols = ["Spearman's R", 'S pval', "Pearson's R", 'P pval', 'RMSE', 'RMSE/std', 'non-zero obs']
    else:
        raise ValueError(task_name)

    # Define result dataframes:
    model_metrics = pd.DataFrame(columns=['matrix', 'polarity'] + main_col_names + score_cols)
    models_predictions = pd.DataFrame()
    counter = 0
    counter_global = 0

    # Loop over training configs:
    model_typename = None
    for train_setup in filenames:
        result_filename = filenames[train_setup]
        loc_models_results = pd.read_csv(results_dir / result_filename, index_col=0)
        loc_models_results = loc_models_results.rename(columns={'Matrix short': 'matrix',
                                                                'Polarity': 'polarity'})

        # Deduce model_typename:
        model_typename = all_model_typenames[np.isin(all_model_typenames, loc_models_results.columns.to_numpy())].item()

        if loc_models_results.prediction.isna().sum() != 0:
            print("NAN VALUES!! {}/{} for {}".format(loc_models_results.prediction.isna().sum(),
                                                     loc_models_results.prediction.shape[0], train_setup),
                  )
            loc_models_results.loc[loc_models_results.prediction.isna(), "prediction"] = 0

        if loc_models_results.observed_value.dtype == "bool":
            loc_models_results.observed_value = np.where(loc_models_results.observed_value, 1, 0)
        if loc_models_results.prediction.dtype == "bool":
            loc_models_results.prediction = np.where(loc_models_results.prediction, 1, 0)
        if task_name == "pytorch_nn_detection" or ("detection" in task_name):
            # Binarize predictions:
            loc_models_results.prediction = np.where(loc_models_results.prediction > 0.5, 1, 0)

        if "regression" in task_name:
            # Remove undetected from regression score:
            loc_models_results = loc_models_results[loc_models_results['observed_value'] != 0]
        elif "rank" in task_name:
            # Remove nan and not-detected from score computation:
            loc_models_results = loc_models_results[loc_models_results['observed_value'] > 0]

        loc_models_results["train_setup"] = train_setup
        models_predictions = pd.concat([models_predictions, loc_models_results])

        # Loop over adduct setups:
        for adduct_setup in adduct_setups_settings:
            # print(adduct_setup)
            # Filter results according to the adduct setup:
            used_adducts = adduct_setups_settings[adduct_setup]["adducts"]
            # print(f"Kept {np.isin(loc_models_results.adduct, used_adducts).sum()} rows out of {loc_models_results.shape}")

            if "adduct" in loc_models_results.columns.to_list():
                loc_models_results_adduct_setup = loc_models_results[np.isin(loc_models_results.adduct, used_adducts)]
            else:
                loc_models_results_adduct_setup = loc_models_results.copy()
            if adduct_setups_settings[adduct_setup]["only_zero_formal_charge"]:
                # print("Removing formal charge...")
                loc_models_results_adduct_setup = loc_models_results[
                    np.isin(loc_models_results.name_short, molecules_with_zero_charge)]

            # Loop over matrices and model types:
            groupby_cols = ['matrix', 'polarity', model_typename]
            if eval_feat_selection: groupby_cols += ['feat_sel_method', 'feat_sel_quantile']
            if has_multiple_iterations: groupby_cols += ['iter_index']
            if load_multiple_thresholds: groupby_cols += [thresh_column_name]
            sum_shape = 0

            for groupby_items, rows in loc_models_results_adduct_setup.groupby(groupby_cols):
                # Compute scores:
                sum_shape += rows.shape[0]
                matrix, polarity, model_type = groupby_items[:3]
                # print(train_setup, groupby_items, rows.shape)
                scores = get_scores(rows, task_type=task_type)

                # Prepare new row of database with scores:
                new_result_row = [matrix, polarity, train_setup, model_type, adduct_setup]
                if eval_feat_selection:
                    feat_sel_met, quantile = groupby_items[3:]
                    new_result_row += [feat_sel_met, round(quantile * 6), rows["nb_features"].drop_duplicates()[0]]
                elif load_multiple_thresholds or has_multiple_iterations:
                    extra_item = groupby_items[3]
                    new_result_row += [extra_item]

                model_metrics.loc[counter] = new_result_row + scores
                counter += 1

            # Compute global scores:
            # Loop over model types:
            groupby_cols = [model_typename]
            if eval_feat_selection: groupby_cols += ['feat_sel_method', 'feat_sel_quantile']
            if load_multiple_thresholds: groupby_cols += [thresh_column_name]
            if has_multiple_iterations: groupby_cols += ['iter_index']
            for groupby_items, rows in loc_models_results_adduct_setup.groupby(groupby_cols):
                # Compute global scores:
                # print(train_setup, groupby_items, rows.shape)
                model_type = groupby_items[0] if (
                            eval_feat_selection or load_multiple_thresholds or has_multiple_iterations) else groupby_items
                scores = get_scores(rows, task_type=task_type)

                # Prepare new row of database with scores:
                new_result_row = [name_global_column, "", train_setup, model_type, adduct_setup]
                if eval_feat_selection:
                    feat_sel_met, quantile = groupby_items[1:]
                    new_result_row += [feat_sel_met, round(quantile * 6), rows["nb_features"].drop_duplicates()[0]]
                elif load_multiple_thresholds or has_multiple_iterations:
                    extra_item = groupby_items[1]
                    new_result_row += [extra_item]

                # model_metrics_global.loc[counter_global] = new_result_row + scores
                model_metrics.loc[counter] = new_result_row + scores
                counter += 1

            # Compute global scores for positive and negative polarities separately:
            groupby_cols += ["polarity"]
            for groupby_items, rows in loc_models_results_adduct_setup.groupby(groupby_cols):
                # Compute global scores:
                model_type = groupby_items[0]
                polarity = groupby_items[-1]
                scores = get_scores(rows, task_type=task_type)

                # Prepare new row of database with scores:
                new_result_row = [name_global_column, polarity, train_setup, model_type, adduct_setup]
                if eval_feat_selection:
                    feat_sel_met, quantile = groupby_items[1:-1]
                    new_result_row += [feat_sel_met, round(quantile * 6), rows["nb_features"].drop_duplicates()[0]]
                elif load_multiple_thresholds or has_multiple_iterations:
                    extra_item = groupby_items[1]
                    new_result_row += [extra_item]

                # model_metrics_global.loc[counter_global] = new_result_row + scores
                model_metrics.loc[counter] = new_result_row + scores
                counter += 1

    if not test_all_adduct_setups:
        # TODO: remove extra duplicates
        pass

    return model_metrics, models_predictions, model_typename
