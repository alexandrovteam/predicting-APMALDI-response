#!/usr/bin/env python
# coding: utf-8

# # Regression spotting project

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.model_selection
from matplotlib import pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

plt.style.use('dark_background')
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

# --------------------------------------
# DEFINE TRAINING FUNCTIONS AND MODELS:
# --------------------------------------

sets_of_models = {
    "regressor":
        {
            'Lin_reg': LinearRegression(),
            'Lin_regMultiOut': LinearRegression(),
            'SVR_rbf': SVR(kernel='rbf', C=100, gamma='auto'),
            # 'SVR_lin': SVR(kernel='linear', C=100, gamma='auto'), # This works terribly
            'SVR_poly': SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1),
            'KNeighbors': KNeighborsRegressor(n_neighbors=5),
            'DecisionTree': DecisionTreeRegressor(max_depth=5),
            'DecisionTreeMultiOut': DecisionTreeRegressor(max_depth=5),
            'RandomForest': RandomForestRegressor(max_depth=5, n_estimators=10),
            'RandomForestMultiOut': RandomForestRegressor(max_depth=5, n_estimators=10),
            'MLP': MLPRegressor(max_iter=1000),
            'MLPMultiOut': MLPRegressor(max_iter=1000),
            'GaussianProcess': GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel()),
            'GaussianProcessMultiOut': GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel())
        },
    "classifier":
        {
            'Logistic_reg': LogisticRegression(),
            'SVC_rbf': SVC(kernel='rbf', C=100, gamma='auto'),
            'SVC_poly': SVC(kernel='poly', C=100, gamma='auto', degree=3, coef0=1),
            'KNeighbors': KNeighborsClassifier(n_neighbors=5),
            'DecisionTree': DecisionTreeClassifier(max_depth=5),
            'DecisionTreeMultiOut': DecisionTreeClassifier(max_depth=5),
            'RandomForest': RandomForestClassifier(max_depth=5, n_estimators=10),
            'RandomForestMultiOut': RandomForestClassifier(max_depth=5, n_estimators=10),
            'MLP': MLPClassifier(max_iter=1000),
            'MLPMultiOut': MLPClassifier(max_iter=1000),
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


def train_one_model_per_matrix_polarity(features_normalized,

                                        intensity_column="spot_intensity",
                                        type_of_models="regressor",
                                        train_loop_function=train_multiple_models,
                                        test_split_col_name='mol_strat_class',
                                        oversampler=None,
                                        test_baseline=False,
                                        feature_selection=False,
                                        use_adduct_features=False,
                                        train_only_on_detected=False
                                        ):
    """

    :param features_normalized:
    :param intensity_column:
    :param type_of_models:
    :param train_loop_function:
    :param test_split_col_name:
    :param oversampler:
    :param test_baseline: Set all input features to zero
    :return:
    """
    # Cross-validation loop:
    regression_results = pd.DataFrame()

    if use_adduct_features:
        pbar_matrices = tqdm(intensities.groupby(by=["matrix", "polarity"]), leave=False)
    else:
        pbar_matrices = tqdm(max_intesities_per_mol.groupby(by=["matrix", "polarity"]), leave=False)
    for (matrix, polarity), rows in pbar_matrices:
        rows = rows.reset_index(drop=True)
        # if train_only_on_detected:
        detected_mask = rows[intensity_column] > 0
        detected_rows = rows[detected_mask].reset_index(drop=True)
        not_detected_rows = rows[~detected_mask].reset_index(drop=True)
        # else:
        pbar_cross_split_detected = tqdm(skf.split(detected_rows.index, detected_rows[test_split_col_name]),
                                leave=False, total=NUM_SPLITS)
        cross_split_not_detected = skf.split(not_detected_rows.index, not_detected_rows[test_split_col_name])

        for fold, ((train_index_det, test_index_det), (train_index_not_det, test_index_not_det)) \
                in enumerate(zip(pbar_cross_split_detected, cross_split_not_detected)):
            # pbar_cross_split.set_postfix({'Cross-validation split': r})

            train_intensities = detected_rows.loc[train_index_det]
            test_intensities = pd.concat([detected_rows.loc[test_index_det], not_detected_rows.loc[test_index_not_det]])
            if not train_only_on_detected:
                train_intensities = pd.concat([train_intensities, not_detected_rows.loc[train_index_not_det]])

            train_y = train_intensities[intensity_column].to_numpy()
            test_y = test_intensities[intensity_column].to_numpy()

            # print(train_intensities[adducts_columns])

            train_cols = adducts_columns.tolist() if use_adduct_features else []
            test_cols = adducts_columns.tolist() + ["adduct"] if use_adduct_features else []
            mol_names_cols = ["name_short", "adduct"] if use_adduct_features else ["name_short"]

            train_x = \
                pd.merge(train_intensities[train_cols + ["name_short"]],
                         features_normalized,
                         how="left",
                         right_index=True,
                         left_on="name_short"
                         ).drop(columns=["name_short"]).to_numpy()
            test_x = \
                pd.merge(test_intensities[test_cols + ["name_short"]],
                         features_normalized,
                         how="left",
                         right_index=True,
                         left_on="name_short"
                         )

            test_mol_names = test_x[mol_names_cols].reset_index(drop=True)
            test_x = test_x.drop(columns=mol_names_cols).to_numpy()

            if test_baseline:
                test_x[:] = 0
                train_x[:] = 0

            if oversampler is not None:
                train_x, train_y = oversampler.fit_resample(train_x, train_y)

            if feature_selection:
                fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
                # ax_list = axes.flat

                X_train_fs, X_test_fs, fs = select_features(train_x, train_y, test_x)
                # what are scores for the features
                for i in range(len(fs.scores_)):
                    print('Feature %d: %f' % (i, fs.scores_[i]))
                # plot the scores
                axes.bar([i for i in range(len(fs.scores_))], fs.scores_)
                plt.show()
                fig.savefig(plots_dir / "selected_feat.pdf")
                print("")

            results_df = train_loop_function(
                train_x, test_x, train_y, test_y,
                type_of_models=type_of_models,
                name_test=test_mol_names,
                test_multioutout_models=False,
                y_is_multioutput=False,
                train=True)

            results_df["matrix"] = matrix
            results_df["polarity"] = polarity
            results_df["fold"] = int(fold)
            regression_results = pd.concat([regression_results, results_df])
    return regression_results


# TODO: generalize to classification
def train_multi_output_regressors(features_normalized, intensity_column="spot_intensity",
                                  test_split_col_name='mol_strat_class'):
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


# feature selection
def select_features(X_train, y_train, X_test, k='all'):
    fs = SelectKBest(score_func=mutual_info_classif, k=k)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# ----------------------------
# LOAD AND NORMALIZE DATA:
# ----------------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, default="regression_on_all")
args = parser.parse_args()
# print(args.accumulate(args.integers))

# TASK_TYPE = "regression_on_all"
# TASK_TYPE = "detection"
TASK_TYPE = args.task_type
# TASK_TYPE = "intensity_classification"

# Paths:
input_dir = Path.cwd() / "../input_data"
plots_dir = Path.cwd() / "../plots"
plots_dir.mkdir(exist_ok=True)
result_dir = Path.cwd() / "../results"
result_dir.mkdir(exist_ok=True, parents=True)

# ## Loading data
# Loading fingerprints, and molecule properties

# Load fingerprints:
fingerprints = pd.read_csv(input_dir / "fingerprints.csv", index_col=0)
fingerprints.sort_index(inplace=True)
# There seems to be some duplicates in the rows:
fingerprints.drop_duplicates(inplace=True)
# Save columns names:
fingerprints_cols = fingerprints.columns

# Load properties:
mol_properties = pd.read_csv(input_dir / "physchem_properties.csv", index_col=0)
mol_properties.sort_index(inplace=True)
mol_properties.drop_duplicates(inplace=True)
# mol_properties.set_index("name_short", inplace=True)
mol_properties_cols = mol_properties.columns

# Check for NaN values:
# FIXME: temporarely set NaN to zero
is_null = mol_properties.isnull()
mol_properties[is_null] = 0.

# Perform some basic checks:
assert fingerprints.index.is_unique
assert mol_properties.index.is_unique

print("Number of fingerprints: ", len(fingerprints))
print("Number of mol properties: ", len(mol_properties))

print("Molecules with missing fingerprints: ")
missing_molecules = list(mol_properties[~ mol_properties.index.isin(fingerprints.index)].index)
print(missing_molecules)

# Intensities:
intensities = pd.read_csv(input_dir / "3june22_ions_no_nl.csv", index_col=0)
intensities = intensities.rename(columns={"Matrix short": "matrix", "Polarity": "polarity"})

intensities.loc[intensities["spot_intensity"] < 100, "spot_intensity"] = 0
intensities.loc[intensities["spot_intensity"] < 100, "detected"] = 0

# Sanity checks:
nb_before = len(intensities.name_short.unique())

# Delete molecules with missing properties:
intensities = intensities[~intensities.name_short.isin(missing_molecules)]
print("{}/{} molecules kept".format(len(intensities.name_short.unique()), nb_before))

remove_not_detected_adducts = False

# if remove_not_detected_adducts:
#     g = intensities.groupby(["name_short", "adduct"], as_index=False)["detected"].max()
#     nb_comb_before = intensities.shape[0]
#     intensities = intensities.merge(g[g["detected"] == 1][["name_short", "adduct"]])
#     print("{}/{} combinations of molecules/adduct with non-zero observed values".format(intensities.shape[0],
#                                                                                         nb_comb_before))
#
#     # Now check if some molecules are never observed (for any adduct) and
#     # remove them from the feature vectors:
#     nb_mol_before = len(all_mol_features.index)
#     all_mol_features = all_mol_features[all_mol_features.index.isin(intensities["name_short"].unique())]
#     print("{}/{} molecules with non-zero observed values".format(len(all_mol_features.index), nb_mol_before))

# #### How many molecules-adduct observed per matrix-polarity
#
# Convert adducts to one-hot encoding:
adducts_one_hot = pd.get_dummies(intensities.adduct, prefix='adduct')
adducts_columns = adducts_one_hot.columns
intensities = intensities.merge(right=adducts_one_hot, right_index=True, left_index=True)

# ## Methods for standartization/normalization
# First, normalize features


ss = StandardScaler()
pt = PowerTransformer()

# OPTION 1
mol_properties_norm_df = pd.DataFrame(pt.fit_transform(mol_properties),
                                      index=mol_properties.index,
                                      columns=mol_properties.columns)

features_norm_df = pd.merge(mol_properties_norm_df, fingerprints, how="inner", right_index=True, left_index=True)

# features_norm_df = pd.DataFrame(pt.fit_transform(all_mol_features),
#                                 index=all_mol_features.index,
#                                 columns=all_mol_features.columns)


# OPTION 2
# features_norm_df = pd.DataFrame(ss.fit_transform(all_mol_features), index = all_mol_features.index, columns = all_mol_features.columns)

# OPTION 3 (Seurat normalization)
# features_norm_df = np.log2((all_mol_features.T / all_mol_features.T.sum().values) * 10000 + 1).T

# #### Intensities normalization


# V2:
numpy_intensities = intensities[["spot_intensity"]].to_numpy()
intensities["norm_intensity_seurat"] = np.log2((numpy_intensities.T / numpy_intensities.T.sum()) * 10000 + 1).T

# Digitize intensities into four classes (low, medium, high, very high):
# Make sure to set noisy predictions (<100) to zero:
intensities.loc[intensities["spot_intensity"] < 100, "norm_intensity_seurat"] = 0
# Now digitize intensities that were normalized with Seurat:
zero_mask = intensities["norm_intensity_seurat"] == 0
intensities.loc[~zero_mask, "norm_intensity_seurat"].hist(bins=4)
_, bins = np.histogram(intensities["norm_intensity_seurat"], bins=4)
bins[-1] += 0.1  # Make sure to include the last point in the last bin
intensities["digitized_seurat"] = np.digitize(intensities["norm_intensity_seurat"], bins=bins)
# Mask not detected intensities:
intensities.loc[zero_mask, "digitized_seurat"] = 0

# Get max intensities across adducts:
max_intesities_per_mol = intensities.groupby(["name_short", "matrix", "polarity"], as_index=False)[
    "digitized_seurat"].max()

# ----------------------------
# CREATE TRAIN/VAL SPLIT:
# ----------------------------

# Since not all the bins have enough datapoints, use quantiles to define the size of the bins:

# We only select only some features, otherwise there are not enough data in each of the splits:
selected_stratification_features = [
    "pka_strongest_basic",
    "polar_surface_area",
    "polarizability"
]
# selected_stratification_features = mol_properties_cols

digitized_mol_properties = pd.DataFrame(index=features_norm_df.index)
for col in selected_stratification_features:
    digitized_mol_properties[col] = pd.qcut(features_norm_df[col], q=2, labels=[1, 2])

digitized_mol_properties['mol_strat_class'] = digitized_mol_properties.astype(str).sum(axis=1).astype('category')


def get_strat_classes(intesity_df, intensity_column="digitized_seurat"):
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
    strat_df['stratification_class'] = strat_df[["mol_strat_class", "matrix_class", intensity_column]].applymap(
        str).apply(''.join, axis=1).astype(
        "category")

    return strat_df['stratification_class']


max_intesities_per_mol['detected'] = (max_intesities_per_mol['digitized_seurat'] > 0).astype("int")

if TASK_TYPE == "detection":
    max_intesities_per_mol['stratification_class'] = get_strat_classes(max_intesities_per_mol, "detected")
    intensities['stratification_class'] = get_strat_classes(intensities, "detected")
elif TASK_TYPE == "intensity_classification":
    max_intesities_per_mol['stratification_class'] = get_strat_classes(max_intesities_per_mol, "digitized_seurat")
    intensities['stratification_class'] = get_strat_classes(intensities, "digitized_seurat")
elif TASK_TYPE == "regression_on_detected":
    # max_intesities_per_mol['stratification_class'] = get_strat_classes(max_intesities_per_mol, "digitized_seurat")
    intensities['stratification_class'] = intensities.merge(digitized_mol_properties,
                                                            left_on="name_short",
                                                            right_index=True,
                                                            how="left")["mol_strat_class"]
elif TASK_TYPE == "regression_on_all":
    intensities['stratification_class'] = get_strat_classes(intensities, "detected")

# ----------------------------
# START TRAINING:
# ----------------------------

# Define cross-validation objects:
NUM_SPLITS = 10
skf = sklearn.model_selection.StratifiedKFold(n_splits=NUM_SPLITS)
skf.get_n_splits()

# Now, train regressors using:
# - Only fingerprints
# - Only mol features
# - Both mol features and fingerprints

# All features:
import time

if TASK_TYPE == "regression_on_all" or TASK_TYPE == "regression_on_detected":
    regr_out = result_dir / TASK_TYPE
    regr_out.mkdir(exist_ok=True, parents=True)

    # Filter intensities, using only detected values:
    # if TASK_TYPE == "regression_on_detected":
    #     intensities = intensities[intensities["detected"] == 1]
    #
    tick = time.time()
    print("Both features")
    regression_results_all_feat = \
        train_one_model_per_matrix_polarity(features_norm_df,
                                            intensity_column="norm_intensity_seurat",
                                            type_of_models="regressor",
                                            test_split_col_name="stratification_class",
                                            use_adduct_features=True,
                                            train_only_on_detected=(TASK_TYPE == "regression_on_detected")
                                            )
    regression_results_all_feat.to_csv(regr_out / "regr_results_all_feat.csv")
    print('Took {} s'.format(time.time() - tick))


    # tick = time.time()
    # print("Uniformly distributed feature")
    # random_features = features_norm_df[mol_properties_cols[[1]]]
    # random_features.iloc[:, :] = np.random.normal(size=random_features.shape)
    # regression_results_all_feat = \
    #     train_one_model_per_matrix_polarity(random_features,
    #                                         intensity_column="norm_intensity_seurat",
    #                                         type_of_models="regressor",
    #                                         test_split_col_name="stratification_class",
    #                                         use_adduct_features=True,
    #                                         train_only_on_detected=(TASK_TYPE == "regression_on_detected")
    #                                         )
    # regression_results_all_feat.to_csv(regr_out / "regr_results_random_feat.csv")
    # print('Took {} s'.format(time.time() - tick))
    #
    #
    # tick = time.time()
    # print("No features")
    # random_features = features_norm_df[mol_properties_cols[[1]]]
    # random_features.iloc[:, :] = np.zeros(shape=random_features.shape, dtype="float")
    # regression_results_all_feat = \
    #     train_one_model_per_matrix_polarity(random_features,
    #                                         intensity_column="norm_intensity_seurat",
    #                                         type_of_models="regressor",
    #                                         test_split_col_name="stratification_class",
    #                                         use_adduct_features=True,
    #                                         train_only_on_detected=(TASK_TYPE == "regression_on_detected")
    #                                         )
    # regression_results_all_feat.to_csv(regr_out / "regr_results_no_feat.csv")
    # print('Took {} s'.format(time.time() - tick))
    #
    #
    # tick = time.time()
    # print("Mol features")
    # regression_results_all_feat = \
    #     train_one_model_per_matrix_polarity(features_norm_df[mol_properties_cols],
    #                                         intensity_column="norm_intensity_seurat",
    #                                         type_of_models="regressor",
    #                                         test_split_col_name="stratification_class",
    #                                         use_adduct_features=True,
    #                                         train_only_on_detected=(TASK_TYPE == "regression_on_detected")
    #                                         )
    # regression_results_all_feat.to_csv(regr_out / "regr_results_mol_feat.csv")
    # print('Took {} s'.format(time.time() - tick))
    #
    # tick = time.time()
    # print("Fingerprints features")
    # regression_results_all_feat = \
    #     train_one_model_per_matrix_polarity(features_norm_df[fingerprints_cols],
    #                                         intensity_column="norm_intensity_seurat",
    #                                         type_of_models="regressor",
    #                                         test_split_col_name="stratification_class",
    #                                         use_adduct_features=True,
    #                                         train_only_on_detected=(TASK_TYPE == "regression_on_detected")
    #                                         )
    # regression_results_all_feat.to_csv(regr_out / "regr_results_fingerprints_feat.csv")
    # print('Took {} s'.format(time.time() - tick))


elif TASK_TYPE == "detection":
    det_out = result_dir / "detection_per_mol"
    det_out.mkdir(parents=True, exist_ok=True)

    # Discretize the intensity:
    max_intesities_per_mol['detected'] = (max_intesities_per_mol['digitized_seurat'] > 0).astype("int")

    # Get oversampler:
    sampler = RandomOverSampler(sampling_strategy="not majority", random_state=43)

    # tick = time.time()
    # print("Both features")
    # regression_results_all_feat = \
    #     train_one_model_per_matrix_polarity(features_norm_df,
    #                                         intensity_column="detected",
    #                                         type_of_models="classifier",
    #                                         test_split_col_name="stratification_class",
    #                                         oversampler=sampler
    #                                         )
    # regression_results_all_feat.to_csv(det_out / "detection_results_all_feat.csv")
    # print('Took {} s'.format(time.time() - tick))
    #
    # tick = time.time()
    # print("Mol features")
    # regression_results_all_feat = \
    #     train_one_model_per_matrix_polarity(features_norm_df[mol_properties_cols],
    #                                         intensity_column="detected",
    #                                         type_of_models="classifier",
    #                                         test_split_col_name="stratification_class",
    #                                         oversampler=sampler
    #                                         )
    # regression_results_all_feat.to_csv(det_out / "detection_results_mol_feat.csv")
    # print('Took {} s'.format(time.time() - tick))
    #
    # tick = time.time()
    # print("Fingerprints features")
    # regression_results_all_feat = \
    #     train_one_model_per_matrix_polarity(features_norm_df[fingerprints_cols],
    #                                         intensity_column="detected",
    #                                         type_of_models="classifier",
    #                                         test_split_col_name="stratification_class",
    #                                         oversampler=sampler,
    #                                         feature_selection=True
    #                                         )
    # regression_results_all_feat.to_csv(det_out / "detection_results_fingerprints_feat.csv")
    # print('Took {} s'.format(time.time() - tick))
    #
    tick = time.time()
    print("One single mol_feat")
    random_features = features_norm_df[mol_properties_cols[[1]]]
    random_features.iloc[:, :] = np.random.normal(size=random_features.shape)
    regression_results_all_feat = \
        train_one_model_per_matrix_polarity(random_features,
                                            intensity_column="detected",
                                            type_of_models="classifier",
                                            test_split_col_name="stratification_class",
                                            oversampler=sampler
                                            )
    regression_results_all_feat.to_csv(det_out / "detection_results_random_feat.csv")
    print('Took {} s'.format(time.time() - tick))

    # tick = time.time()
    # print("No features")
    # zero_feat = features_norm_df[mol_properties_cols[[1]]]
    # zero_feat.iloc[:, :] = 0
    # regression_results_all_feat = \
    #     train_one_model_per_matrix_polarity(zero_feat,
    #                                         intensity_column="detected",
    #                                         type_of_models="classifier",
    #                                         test_split_col_name="stratification_class",
    #                                         oversampler=sampler,
    #                                         test_baseline=True
    #                                         )
    # regression_results_all_feat.to_csv(det_out / "detection_results_no_features.csv")
    # print('Took {} s'.format(time.time() - tick))


elif TASK_TYPE == "intensity_classification":
    det_out = result_dir / "intensity_classification_per_mol"
    det_out.mkdir(parents=True, exist_ok=True)

    # Get oversampler:
    sampler = RandomOverSampler(sampling_strategy="not majority", random_state=43)

    tick = time.time()
    print("Both features")
    regression_results_all_feat = \
        train_one_model_per_matrix_polarity(features_norm_df,
                                            intensity_column="digitized_seurat",
                                            type_of_models="classifier",
                                            test_split_col_name="stratification_class",
                                            oversampler=sampler
                                            )
    regression_results_all_feat.to_csv(det_out / "intensity_classification_all_feat.csv")
    print('Took {} s'.format(time.time() - tick))

    tick = time.time()
    print("Mol features")
    regression_results_all_feat = \
        train_one_model_per_matrix_polarity(features_norm_df[mol_properties_cols],
                                            intensity_column="digitized_seurat",
                                            type_of_models="classifier",
                                            test_split_col_name="stratification_class",
                                            oversampler=sampler
                                            )
    regression_results_all_feat.to_csv(det_out / "intensity_classification_mol_feat.csv")
    print('Took {} s'.format(time.time() - tick))

    tick = time.time()
    print("Fingerprints features")
    regression_results_all_feat = \
        train_one_model_per_matrix_polarity(features_norm_df[fingerprints_cols],
                                            intensity_column="digitized_seurat",
                                            type_of_models="classifier",
                                            test_split_col_name="stratification_class",
                                            oversampler=sampler
                                            )
    regression_results_all_feat.to_csv(det_out / "intensity_classification_fingerprints_feat.csv")
    print('Took {} s'.format(time.time() - tick))

    tick = time.time()
    print("One random features")
    regression_results_all_feat = \
        train_one_model_per_matrix_polarity(features_norm_df[mol_properties_cols[[1]]],
                                            intensity_column="digitized_seurat",
                                            type_of_models="classifier",
                                            test_split_col_name="stratification_class",
                                            oversampler=sampler
                                            )
    regression_results_all_feat.to_csv(det_out / "intensity_classification_one_random_mol_feat.csv")
    print('Took {} s'.format(time.time() - tick))
else:
    raise NotImplementedError(f"Task type not recognized {TASK_TYPE}")
