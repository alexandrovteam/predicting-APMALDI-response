from pathlib import Path

import numpy as np
import pandas as pd
# import shap

import sklearn.model_selection
from matplotlib import pyplot as plt

from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression

from predicting_APMALDI_response.train_pytorch_models import train_pytorch_model_wrapper

# plt.style.use('dark_background')
from tqdm import tqdm

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import argparse

sets_of_models = {
    "regressor":
        {
            # 'Lin_reg': LinearRegression(),
            # # 'Lin_regMultiOut': LinearRegression(),
            # 'SVR_rbf': SVR(kernel='rbf', C=100, gamma='auto'),
            # 'SVR_lin': SVR(kernel='linear', C=100, gamma='auto'), # This works terribly
            # 'SVR_poly': SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1),
            # 'KNeighbors': KNeighborsRegressor(n_neighbors=5),
            # 'DecisionTree': DecisionTreeRegressor(max_depth=5),
            # # 'DecisionTreeMultiOut': DecisionTreeRegressor(max_depth=5),
            # 'RandomForest': RandomForestRegressor(max_depth=5, n_estimators=10),
            # # 'RandomForestMultiOut': RandomForestRegressor(max_depth=5, n_estimators=10),
            'MLP': MLPRegressor(max_iter=2000),
            # # 'MLPMultiOut': MLPRegressor(max_iter=2000),
            # 'GaussianProcess': GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel()),
            # # 'GaussianProcessMultiOut': GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel())
        },

    "classifier":
        {
            # 'Logistic_reg': LogisticRegression(),
            # # 'SVC_rbf': SVC(kernel='rbf', C=100, gamma='auto'),
            # 'SVC_poly': SVC(kernel='poly', C=100, gamma='auto', degree=3, coef0=1),
            # 'KNeighbors': KNeighborsClassifier(n_neighbors=5),
            # 'DecisionTree': DecisionTreeClassifier(max_depth=5),
            # # 'DecisionTreeMultiOut': DecisionTreeClassifier(max_depth=5),
            # 'RandomForest': RandomForestClassifier(max_depth=5, n_estimators=10),
            # 'RandomForestMultiOut': RandomForestClassifier(max_depth=5, n_estimators=10),
            'MLP': MLPClassifier(max_iter=2000),
            # 'MLPMultiOut': MLPClassifier(max_iter=2000),
            # 'GaussianProcess': GaussianProcessClassifier(kernel=DotProduct() + WhiteKernel()),
            # 'GaussianProcessMultiOut': GaussianProcessClassifier(kernel=DotProduct() + WhiteKernel())
        }
}


def train_multiple_models(train_x, test_x, train_y, test_y,
                          type_of_models="regressor",
                          out_multi_index=None,
                          model_set=(),
                          name_test=None,
                          train=True,
                          test_multioutout_models=True,
                          y_is_multioutput=True
                          ):
    if y_is_multioutput: assert out_multi_index is not None

    results_df = pd.DataFrame()
    all_models = sets_of_models[type_of_models]

    if len(model_set) == 0: model_set = all_models.keys()

    if name_test is not None and y_is_multioutput:
        nb_outputs = test_y.shape[1]
        name_test = name_test.loc[name_test.index.repeat(nb_outputs)]

    pbar = tqdm(model_set, leave=False)
    # for r in model_set:
    for r in pbar:
        pbar.set_postfix({type_of_models: r})
        regressor = all_models[r]
        # These classifiers need to train several models (one per class):
        if "MultiOut" not in r and y_is_multioutput:
            regressor = MultiOutputRegressor(regressor)
        elif "MultiOut" in r and (not test_multioutout_models or not y_is_multioutput):
            # Skip MultiOutput models:
            continue
        if train:
            regressor.fit(train_x, train_y)
            y_pred = regressor.predict(test_x)

            if y_is_multioutput:
                loc_res_df = pd.DataFrame(test_y, columns=out_multi_index).stack(
                    [i for i in range(len(out_multi_index.levels))]).reset_index().drop(columns=["level_0"]).rename(
                    columns={0: "observed_value"})
                loc_res_df['prediction'] = pd.DataFrame(y_pred, columns=out_multi_index).stack(
                    [i for i in range(len(out_multi_index.levels))]).reset_index()[0]
                loc_res_df[type_of_models] = r
                loc_res_df.reset_index(drop=True, inplace=True)
                if name_test is not None:
                    loc_res_df = loc_res_df.merge(name_test.reset_index(drop=True), left_index=True, right_index=True)
            else:
                loc_res_df = pd.DataFrame({'observed_value': test_y,
                                           'prediction': y_pred,
                                           type_of_models: r})
                if name_test is not None:
                    loc_res_df = loc_res_df.merge(name_test, left_index=True, right_index=True)
        else:
            raise NotImplementedError()
        results_df = pd.concat([results_df, loc_res_df])

    results_df = results_df.reset_index(drop=True)
    return results_df


def select_important_features(X, Y, feat_names,
                              task_type="regression",
                              feature_type="numerical"):
    """

    :param X:
    :param Y:
    :param feat_names:
    :param task_type:
    :param feature_type:
    :return:
    """
    assert task_type in ["regression", "classification"], f"Task type {task_type} not supported"
    assert feature_type in ["numerical", "categorical"], f"Feature type {feature_type} not suppported"

    if task_type != "regression": raise NotImplementedError("Classification is not supported yet")

    mdl_important_features = pd.DataFrame(index=feat_names)

    if task_type == "classification":
        rf = RandomForestRegressor(max_depth=5, n_estimators=10)
        # rf = RandomForestClassifier(max_depth=5, n_estimators=10)
        mlp = MLPClassifier(max_iter=1000)
    elif task_type == "regression":
        rf = RandomForestRegressor(max_depth=5, n_estimators=10)
        mlp = MLPRegressor(max_iter=1000)
    else:
        raise ValueError(task_type)

    # -----------------------
    # Random Forest feat selection:
    # -----------------------
    if feature_type == "categorical":
        rf.fit(X, Y)
        mdl_important_features.loc[feat_names, 'RandomForest'] = rf.feature_importances_

    # -----------------------
    # MLP Feat selection:
    # -----------------------

    # mlp.fit(X, Y)
    #
    # # explainer_clf = shap.KernelExplainer(mlp.predict, X)
    # explainer_clf = shap.DeepExplainer(mlp.predict, X)
    # shap_values_clf = explainer_clf.shap_values(X)
    #
    # shap.summary_plot(shap_values=shap_values_clf, features=X,
    #                   feature_names=feat_names, class_names=['not detected', 'detected'],
    #                   color_bar_label='Feature level',
    #                   max_display=40,
    #                   show=False
    #                   )
    # plt.tight_layout()
    # plt.show()

    # -----------------------
    # Other general feature selection methods:
    # -----------------------
    X_df, Y_df = pd.DataFrame(X), pd.Series(Y)
    if feature_type == "numerical":
        # # Mutual information
        fi_clf = mutual_info_regression(X, Y)
        mdl_important_features['mutual_info'] = fi_clf

        # # Polynomial fit
        polifit_res = np.polyfit(Y, X, 2, full=True)
        mdl_important_features['polyfit'] = polifit_res[1]

        # Correlation
        # pearson = X_df.corrwith(Y_df, method='pearson')
        # mdl_important_features['pearson'] = np.abs(pearson.values)
        spearman = X_df.corrwith(Y_df, method='spearman')
        mdl_important_features['spearman'] = np.abs(spearman.values)

        # # # Linear regression
        # from sklearn.linear_model import LinearRegression
        # lr = LinearRegression()
        # lr.fit(train_x, train_y)
        # important_features_df['Lin reg importance'] = lr.coef_[0]
    elif feature_type == "categorical":
        pass

        # Correlation
        spearman = X_df.corrwith(Y_df, method='kendall')
        mdl_important_features['kendall'] = np.abs(spearman.values)

    return mdl_important_features


def convert_df_to_training_format(intensities_df, feat_df,
                                  gt_column_name,
                                  use_adduct_features=False,
                                  adducts_columns=None):
    col_names = adducts_columns.tolist() + ["adduct"] if use_adduct_features else []
    mol_names_cols = ["name_short", "adduct"] if use_adduct_features else ["name_short"]
    Y = intensities_df[gt_column_name].to_numpy()
    X = \
        pd.merge(intensities_df[col_names + ["name_short"]],
                 feat_df,
                 how="left",
                 right_index=True,
                 left_on="name_short"
                 )
    mol_names_df = X[mol_names_cols].reset_index(drop=True)
    X = X.drop(columns=mol_names_cols).to_numpy()
    return X, Y, mol_names_df


def cross_val_loop(input_df, feat_df, matrix, polarity,
                   intensity_column="spot_intensity",
                   type_of_models="regressor",
                   train_loop_function=train_multiple_models,
                   test_split_col_name='mol_strat_class',
                   oversampler=None,
                   test_baseline=False,
                   train_only_on_detected=False,
                   use_adduct_features=False,
                   num_cross_val_folds=10,
                   adducts_columns=None
                   ):
    """
    Cross validation loop
    """
    results_df = pd.DataFrame()
    # Get cross-validation splits:
    detected_mask = input_df[intensity_column] > 0
    detected_rows = input_df[detected_mask].reset_index(drop=True)
    not_detected_rows = input_df[~detected_mask].reset_index(drop=True)
    skf = sklearn.model_selection.StratifiedKFold(n_splits=num_cross_val_folds)
    pbar_cross_split_detected = tqdm(skf.split(detected_rows.index, detected_rows[test_split_col_name]),
                                     leave=False, total=num_cross_val_folds)
    cross_split_not_detected = skf.split(not_detected_rows.index, not_detected_rows[test_split_col_name])

    for fold, ((train_index_det, test_index_det), (train_index_not_det, test_index_not_det)) \
            in enumerate(zip(pbar_cross_split_detected, cross_split_not_detected)):

        pbar_cross_split_detected.set_postfix({"Fold": fold})
        # Get intensities used for test and train:
        train_intensities = detected_rows.loc[train_index_det]
        test_intensities = pd.concat(
            [detected_rows.loc[test_index_det], not_detected_rows.loc[test_index_not_det]])
        if not train_only_on_detected:
            train_intensities = pd.concat([train_intensities, not_detected_rows.loc[train_index_not_det]])

        # Convert data to X, Y format:
        train_x, train_y, _ = convert_df_to_training_format(train_intensities, feat_df,
                                                            intensity_column, use_adduct_features,
                                                            adducts_columns)
        test_x, test_y, test_mol_names = convert_df_to_training_format(test_intensities, feat_df,
                                                                       intensity_column, use_adduct_features,
                                                                       adducts_columns)

        if test_baseline:
            test_x[:] = 0
            train_x[:] = 0

        if oversampler is not None:
            train_x, train_y = oversampler.fit_resample(train_x, train_y)

        # Start training:
        loc_results_df = train_loop_function(
            train_x, test_x, train_y, test_y,
            type_of_models=type_of_models,
            name_test=test_mol_names,
            test_multioutout_models=False,
            y_is_multioutput=False,
            train=True)

        loc_results_df["matrix"] = matrix
        loc_results_df["polarity"] = polarity
        loc_results_df["fold"] = int(fold)
        results_df = pd.concat([results_df, loc_results_df])
    return results_df


def train_one_model_per_matrix_polarity(input_df,
                                        features_normalized,
                                        intensity_column="spot_intensity",
                                        type_of_models="regressor",
                                        train_loop_function=train_multiple_models,
                                        test_split_col_name='mol_strat_class',
                                        oversampler=None,
                                        test_baseline=False,
                                        use_adduct_features=False,
                                        train_only_on_detected=False,
                                        adducts_columns=None,
                                        do_feature_selection=False,
                                        only_save_feat_sel_results=False,
                                        features_type="categorical",
                                        path_feature_importance_csv=None,
                                        num_cross_val_folds=10,
                                        feature_selection_out_dir=None
                                        ):
    """
    :param test_baseline: Set all input features to zero
    """
    model_predictions = pd.DataFrame()

    pbar_matrices = tqdm(input_df.groupby(by=["matrix", "polarity"]), leave=False)

    all_feat_importance = pd.DataFrame()

    # Loop over types of matrices:
    for (matrix, polarity), rows in pbar_matrices:
        pbar_matrices.set_postfix({"Matrix": f"{matrix}, {polarity}"})
        rows = rows.reset_index(drop=True)

        if not do_feature_selection:
            # Simply perform the cross-validation loop using all features:
            model_predictions = pd.concat([model_predictions,
                                           cross_val_loop(rows, features_normalized, matrix, polarity,
                                                          intensity_column, type_of_models,
                                                          train_loop_function, test_split_col_name, oversampler,
                                                          test_baseline, train_only_on_detected,
                                                          use_adduct_features, num_cross_val_folds,
                                                          adducts_columns)])
        else:
            # ----------------------------------------------
            # First, find important features using all data:
            # ----------------------------------------------
            feat_names = features_normalized.columns.tolist()
            if use_adduct_features:
                feat_names = adducts_columns.tolist() + feat_names

            if path_feature_importance_csv is not None:
                # Load feature importance from file:
                mdl_important_features = pd.read_csv(path_feature_importance_csv, index_col=0)
                # Get data for the given matrix/polarity:
                mdl_important_features = \
                    mdl_important_features[(mdl_important_features.matrix == matrix) &
                                           (mdl_important_features.polarity == polarity)]
                mdl_important_features = mdl_important_features.drop(columns={"matrix", "polarity"}).astype('float')
                # for col in mdl_important_features.columns:
                #     mdl_important_features = mdl_important_features[col].astype('float')
            else:
                # Get all data in training format X, Y:
                X_global, Y_global, mol_names = convert_df_to_training_format(rows, features_normalized,
                                                                      intensity_column, use_adduct_features,
                                                                      adducts_columns)

                # TODO: generalize and fix this mess
                # Get feature scores:
                # mdl_important_features = select_important_features(X_global, Y_global, feat_names,
                #                                                    feature_type=features_type)
                mdl_important_features = train_pytorch_model_wrapper(train_x=X_global,
                                                                     train_y=Y_global,
                                                                     feature_names=feat_names,
                                                                     type_of_models=type_of_models,
                                                                     do_feature_selection=True,
                                                                     y_is_multioutput=False,
                                                                     matrix=matrix,
                                                                     polarity=polarity,
                                                                     molecule_names=mol_names,
                                                                     feature_selection_out_dir=feature_selection_out_dir)
                #

            # ----------------------------------------------
            # Next, train models using different quantile thresholds for feature importance:
            # ----------------------------------------------
            # # Mask everything that is below 5% of the range:
            # min_thresh = (mdl_important_features.max() - mdl_important_features.min()) * 0.05
            # mdl_important_features[mdl_important_features < min_thresh] = np.nan

            if only_save_feat_sel_results:
                mdl_important_features["matrix"] = matrix
                mdl_important_features["polarity"] = polarity
                all_feat_importance = pd.concat([all_feat_importance, mdl_important_features])
            else:
                n_thresholds = 5  # NUMBER OF THRESHOLDS, CAN BE CHANGED
                quantiles = np.linspace(0, 1, n_thresholds + 2)  # [:-1]
                feat_quantiles = mdl_important_features.abs().quantile(quantiles)

                pbar_feat_sel_methods = tqdm(mdl_important_features.columns, leave=False)
                for feat_sel_method in pbar_feat_sel_methods:
                    pbar_feat_sel_methods.set_postfix({"Feat-sel method": feat_sel_method})
                    loc_scores_important_feat = mdl_important_features[feat_sel_method]

                    quantiles_tqdm = tqdm(quantiles, leave=False)
                    last_thresh = -1
                    for i, q in enumerate(quantiles_tqdm):
                        q_thresh = feat_quantiles.loc[q, feat_sel_method]
                        if q_thresh == last_thresh:
                            continue
                        last_thresh = q_thresh
                        important_features = loc_scores_important_feat[
                            loc_scores_important_feat >= q_thresh].index.tolist()
                        # Possibly remove adduct feat, that will anyway added in a second moment:
                        important_features = [feat for feat in important_features if
                                              "adduct" not in feat]
                        quantiles_tqdm.set_postfix({"Quantile": (int(q * (n_thresholds + 2)), len(important_features))})
                        # Filter features:
                        if len(important_features) == 0:
                            # Just give a random feature instead:
                            important_feat_df = pd.DataFrame(
                                data=np.random.normal(size=(features_normalized.shape[0], 1)),
                                index=features_normalized.index,
                                columns=["random_feature"]
                            )
                        else:
                            important_feat_df = features_normalized[important_features]
                            # important_feat_array = np.random.features_normalized.iloc[:,0].shape
                        loc_model_predictions = cross_val_loop(rows, important_feat_df,
                                                               matrix, polarity, intensity_column, type_of_models,
                                                               train_loop_function, test_split_col_name, oversampler,
                                                               test_baseline, train_only_on_detected,
                                                               use_adduct_features, num_cross_val_folds,
                                                               adducts_columns)
                        loc_model_predictions["feat_sel_method"] = feat_sel_method
                        loc_model_predictions["feat_sel_quantile"] = q
                        loc_model_predictions["nb_features"] = len(important_features)
                        # print(q, len(important_features))
                        model_predictions = pd.concat([model_predictions, loc_model_predictions])

    if only_save_feat_sel_results and do_feature_selection:
        return all_feat_importance

    return model_predictions


# TODO: generalize to classification
def train_multi_output_regressors(features_normalized, intensity_column="spot_intensity",
                                  test_split_col_name='mol_strat_class'):
    raise DeprecationWarning()
    # Cross-validation loop:
    regression_results = pd.DataFrame()
    selected_mols = digitized_mol_properties

    sorted_intensities = intensities.sort_values(by=['name_short', 'adduct', "matrix", "polarity"])
    sorted_intensities = pd.merge(sorted_intensities,
                                  features_normalized,
                                  how="left",
                                  right_index=True,
                                  left_on="name_short"
                                  )

    # Create folds, making sure that there are enough detected observations for each matrix:

    pbar_cross_split = tqdm(skf.split(selected_mols.index, selected_mols[test_split_col_name]), leave=False,
                            total=NUM_SPLITS)

    for fold, (train_index, test_index) in enumerate(pbar_cross_split):
        train_intensities = sorted_intensities[sorted_intensities.name_short.isin(selected_mols.index[train_index])]
        test_intensities = sorted_intensities[sorted_intensities.name_short.isin(selected_mols.index[test_index])]
        g_train = train_intensities.groupby(by=['name_short', 'adduct'])
        g_test = test_intensities.groupby(by=['name_short', 'adduct'])
        train_y = np.array([rows.to_numpy() for _, rows in g_train[intensity_column]])
        train_x = np.array([rows.iloc[0].to_numpy() for _, rows in
                            g_train[adducts_columns.tolist() + features_normalized.columns.tolist()]])

        test_y = np.array([rows.to_numpy() for _, rows in g_test[intensity_column]])
        test_x = np.array([rows.iloc[0].to_numpy() for _, rows in
                           g_test[adducts_columns.tolist() + features_normalized.columns.tolist()]])

        test_mol_names = pd.DataFrame([[name, adduct] for (name, adduct), _ in g_test],
                                      columns=["name_short", "adduct"])

        matrix_names = train_intensities[["matrix", "polarity"]].drop_duplicates()
        out_multi_index = pd.MultiIndex.from_arrays([matrix_names["matrix"], matrix_names["polarity"]])
        results_df = train_multiple_models(train_x, test_x, train_y, test_y,
                                           type_of_models="regressor",
                                           name_test=test_mol_names,
                                           out_multi_index=out_multi_index,
                                           train=True)
        results_df["fold"] = int(fold)
        regression_results = pd.concat([regression_results, results_df])
    return regression_results


def get_strat_classes(intesity_df, digitized_mol_properties,
                      intensity_column="digitized_seurat",
                      stratify_not_detected=True):
    # Get class depending on matrix:
    strat_df = \
        intesity_df.merge(digitized_mol_properties, left_on="name_short", right_index=True, how="left")[
            [intensity_column, "matrix", "polarity", "mol_strat_class"]]
    strat_df['mol_strat_class'] = strat_df['mol_strat_class'].astype(int)
    strat_df = pd.get_dummies(strat_df, columns=["matrix", "polarity"], prefix=["mat", "pol"])
    strat_df['matrix_class'] = strat_df.iloc[:, 2:].applymap(str).apply(''.join, axis=1)
    # Mask out things that are not detected:
    strat_df.loc[strat_df[intensity_column] == 0, 'matrix_class'] = 0
    # Apply molecule features stratification only to molecules that were not detected:
    strat_df.loc[strat_df[intensity_column] > 0, 'mol_strat_class'] = 0
    # Finally, merge classes:
    strat_cols = ["matrix_class", intensity_column]
    if stratify_not_detected: strat_cols = ["mol_strat_class"] + strat_cols
    strat_df['stratification_class'] = strat_df[strat_cols].applymap(
        str).apply(''.join, axis=1).astype(
        "category")

    return strat_df['stratification_class']
