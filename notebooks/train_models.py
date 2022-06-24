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

plt.style.use('dark_background')

from sklearn.preprocessing import StandardScaler
import umap.plot

# In[145]:


# Paths:
input_dir = Path.cwd() / "../input_data"
plots_dir = Path.cwd() / "../plots"
plots_dir.mkdir(exist_ok=True)
result_dir = Path.cwd() / "../results/multioutput"
result_dir.mkdir(exist_ok=True, parents=True)

# ## Loading data
# Loading fingerprints, and molecule properties
# 

# In[3]:


# Load fingerprints:
fingerprints = pd.read_csv(input_dir / "fingerprints.csv", index_col=0)
fingerprints.sort_index(inplace=True)
# There seems to be some duplicates in the rows:
fingerprints.drop_duplicates(inplace=True)
# Save columns names:
fingerprints_cols = fingerprints.columns

fingerprints
# fingerprints.addu


# In[4]:


# Load properties:
mol_properties = pd.read_csv(input_dir / "physchem_properties.csv", index_col=0)
mol_properties.sort_index(inplace=True)
mol_properties.drop_duplicates(inplace=True)
# mol_properties.set_index("name_short", inplace=True)
mol_properties_cols = mol_properties.columns
mol_properties

# Check statistics of molecular properties and handle NaN values:
# 

# In[5]:


# Plot histograms:
fig = plt.figure(figsize=(20, 20))
ax = fig.gca()
mol_properties.hist(ax=ax)
fig.savefig(plots_dir / "mol_features.pdf")

# In[6]:


# Check for NaN values:
is_null = mol_properties.isnull()
for col in mol_properties_cols:
    print("---")
    print(is_null[col].value_counts())

# FIXME: temporarely set NaN to zero
mol_properties[is_null] = 0.

# In[7]:


# Perform some basic checks:
assert fingerprints.index.is_unique
assert mol_properties.index.is_unique

# In[8]:


print("Number of fingerprints: ", len(fingerprints))
print("Number of mol properties: ", len(mol_properties))

print("Molecules with missing fingerprints: ")
missing_molecules = list(mol_properties[~ mol_properties.index.isin(fingerprints.index)].index)
print(missing_molecules)

# min(fingerprints.iloc[3, ] == fingerprints.iloc[4, ])


# Merge fingerprints and properties:

# In[9]:


all_mol_features = pd.merge(mol_properties, fingerprints, how="inner", right_index=True, left_index=True)

# ### Loading intensities

# In[46]:


# Intensities:
intensities = pd.read_csv(input_dir / "3june22_ions_no_nl.csv", index_col=0)
intensities = intensities.rename(columns={"Matrix short": "matrix", "Polarity": "polarity"})
intensities.head()

# In[47]:


# Sanity checks:
nb_before = len(intensities.name_short.unique())

# Delete molecules with missing properties:
intensities = intensities[~intensities.name_short.isin(missing_molecules)]
print("{}/{} molecules kept".format(len(intensities.name_short.unique()), nb_before))

# #### Remove molecule-adduct combinations that were never observed in any matrix

# In[48]:


remove_not_detected_adducts = False

if remove_not_detected_adducts:
    g = intensities.groupby(["name_short", "adduct"], as_index=False)["detected"].max()
    nb_comb_before = intensities.shape[0]
    intensities = intensities.merge(g[g["detected"] == 1][["name_short", "adduct"]])
    print("{}/{} combinations of molecules/adduct with non-zero observed values".format(intensities.shape[0],
                                                                                        nb_comb_before))

    # Now check if some molecules are never observed (for any adduct) and
    # remove them from the feature vectors:
    nb_mol_before = len(all_mol_features.index)
    all_mol_features = all_mol_features[all_mol_features.index.isin(intensities["name_short"].unique())]
    print("{}/{} molecules with non-zero observed values".format(len(all_mol_features.index), nb_mol_before))

# #### How many molecules-adduct observed per matrix-polarity
# 

# In[49]:


g = intensities.groupby(['matrix', 'polarity'], as_index=False)
detected_ratio = g["detected"].apply(lambda x: x.sum() / x.shape[0])
detected_ratio["Sum detected"] = g["detected"].sum()["detected"]

# Show values:
detected_ratio.sort_values("detected", ascending=False).rename(columns={"detected": "Ratio detected molecules"})

# Get statistics about adducts:
# 

# In[50]:


std_intesities = intensities.groupby(['matrix', "polarity", "name_short"], as_index=False).std()

fig = plt.figure(figsize=(15, 20))
ax = fig.gca()
std_intesities.hist(column=["detected"], by=['matrix', "polarity"], grid=False, ax=ax)
fig.savefig(plots_dir / "std_detected_value.pdf")

# In[51]:


# Check if entries with std=0 are always not detected:
std_intesities[std_intesities["detected"] == 0]["spot_intensity"].value_counts()

# **Conclusion**: Intensities always vary across adducts (in histograms, std=0 entries were never detected for any adducts)

# In[52]:


# Convert adducts to one-hot encoding:
adducts_one_hot = pd.get_dummies(intensities.adduct, prefix='adduct')
adducts_columns = adducts_one_hot.columns
intensities = intensities.merge(right=adducts_one_hot, right_index=True, left_index=True)

# ### How many matrices with non-zero values per molecule+adduct
# Many adducts occure in several matrices, but let's see how intensities look like, ranked by higher to lower:

# In[53]:


g = intensities.sort_values("spot_intensity", ascending=False).groupby(["name_short", "adduct"], as_index=False)
intensity_stats = g["detected"].sum()
for matrix_index in range(20):
    # intensity_stats["matrix_{}".format(matrix_index)] = g["spot_intensity"].max()["spot_intensity"]
    intensity_stats["matrix_{}".format(matrix_index)] = g["spot_intensity"].apply(lambda x: x.iloc[matrix_index])[
        "spot_intensity"]
# assert intensity_stats["max"].equals(intensity_stats["max2"])
intensity_stats

# #### Intensity histograms across matrices (before normalization)
# Only plot intensities > 100:

# In[54]:


fig = plt.figure(figsize=(15, 20))
ax = fig.gca()

_ = intensities[intensities["spot_intensity"] > 100.].hist("spot_intensity", by=["matrix", "polarity"], ax=ax,
                                                           sharex=True)

# In[56]:


_ = intensities[intensities["spot_intensity"] > 100].hist("spot_intensity")

# ## Methods for standartization/normalization
# First, normalize features

# In[57]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

ss = StandardScaler()
pt = PowerTransformer()

# OPTION 1
features_norm_df = pd.DataFrame(pt.fit_transform(all_mol_features),
                                index=all_mol_features.index,
                                columns=all_mol_features.columns)

# OPTION 2
# features_norm_df = pd.DataFrame(ss.fit_transform(all_mol_features), index = all_mol_features.index, columns = all_mol_features.columns)

# OPTION 3 (Seurat normalization)
# features_norm_df = np.log2((all_mol_features.T / all_mol_features.T.sum().values) * 10000 + 1).T


# In[68]:


num_rows = 3
num_cols = 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
axes = [axes[i, j] for i in range(num_cols) for j in range(num_rows)]
for i, ax in enumerate(axes):
    if i < len(mol_properties_cols):
        sc = ax.scatter(all_mol_features[mol_properties_cols[i]], features_norm_df[mol_properties_cols[i]],
                        # cmap = plt.cm.SOME_CM
                        )
        # plt.colorbar(sc, ax=ax)
        ax.set_title(mol_properties_cols[i])
    # ax.axis('off')

# #### Intensities normalization

# In[58]:


# intensities.loc[intensities["spot_intensity"] < 100, "spot_intensity"] = 0

# V1 normalization:
intensities["norm_intensity"] = pt.fit_transform(intensities[["spot_intensity"]])

# V2:
numpy_intensities = intensities[["spot_intensity"]].to_numpy()
intensities["norm_intensity_seurat"] = np.log2((numpy_intensities.T / numpy_intensities.T.sum()) * 10000 + 1).T

# Histograms for normalization_v1: zeros values are mapped to negative values. Everything that originally had intensity < 100 is not plotted

# In[59]:


fig = plt.figure(figsize=(15, 20))
ax = fig.gca()

_ = intensities[intensities["spot_intensity"] > 100].hist("norm_intensity", by=["matrix", "polarity"], ax=ax,
                                                          sharex=True)

# In[64]:


_ = intensities[intensities["norm_intensity"] > 0].hist("norm_intensity", bins=4)

# Histograms for Seurat normalization:

# In[22]:


fig = plt.figure(figsize=(15, 20))
ax = fig.gca()

_ = intensities[intensities["spot_intensity"] > 100].hist("norm_intensity_seurat", by=["matrix", "polarity"], ax=ax,
                                                          sharex=True)

# In[63]:


_ = intensities[intensities["norm_intensity_seurat"] > 0].hist("norm_intensity_seurat", bins=4)

# In[65]:


g = intensities.sort_values("norm_intensity_seurat", ascending=False).groupby(["name_short", "adduct"], as_index=False)
intensity_stats = g["detected"].sum()
for matrix_index in range(20):
    # intensity_stats["matrix_{}".format(matrix_index)] = g["norm_intensity_seurat"].max()["norm_intensity_seurat"]
    intensity_stats["matrix_{}".format(matrix_index)] = \
        g["norm_intensity_seurat"].apply(lambda x: x.iloc[matrix_index])["norm_intensity_seurat"]
# assert intensity_stats["max"].equals(intensity_stats["max2"])
intensity_stats

# Overwrite original intensities values with normalization v1:

# In[66]:


intensities.plot.scatter("spot_intensity", "norm_intensity_seurat")

# In[69]:


# intensities["spot_intensity"] = intensities["norm_intensity_seurat"]


# ## Create train/val split
# 
# First, binarize the molecule features:

# In[30]:


fig = plt.figure(figsize=(20, 20))
ax = fig.gca()
features_norm_df[mol_properties_cols].hist(ax=ax)

# Since not all the bins have enough datapoints, use quantiles to define the size of the bins:

# In[31]:


# We only select only some features, otherwise there are not enough data in each of the splits:
# selected_stratification_features = [
#     "pka_strongest_basic",̊
#     "polar_surface_area",
#     "polarizability"
# ]
selected_stratification_features = mol_properties_cols

digitized_mol_properties = pd.DataFrame(index=features_norm_df.index)
for col in selected_stratification_features:
    digitized_mol_properties[col] = pd.qcut(features_norm_df[col], q=2, labels=[1, 2])

# digitized_mol_properties.value_counts()
# digitized_mol_properties


# Now let's get the product of all the classes used for stratification:

# In[32]:


# In[33]:


# First, remove adduct information from the intensity dataframe:
matrix_pol_df = intensities.drop(columns=['adduct', 'detected',
                                          'spot_intensity']).drop_duplicates().set_index('name_short', )

strat_feat = pd.merge(matrix_pol_df, digitized_mol_properties, how="left", left_index=True,
                      right_index=True)
strat_feat.value_counts()

strat_feat['combined'] = strat_feat.astype(str).sum(axis=1).astype('category')

# In[34]:


digitized_mol_properties['combined'] = digitized_mol_properties.astype(str).sum(axis=1).astype('category')

# digitized_mol_properties['combined'].value_counts()

# Get class depending on matrix:
strat_df = intensities.merge(digitized_mol_properties, left_on="name_short", right_index=True, how="left")[
    ["spot_intensity", "matrix", "polarity", "combined"]]
strat_df['combined'] = strat_df['combined'].astype(int)
strat_df = pd.get_dummies(strat_df, columns=["matrix", "polarity"], prefix=["mat", "pol"])
strat_df['matrix_class'] = strat_df.iloc[:, 2:].applymap(str).apply(''.join, axis=1)
# Mask out things that are not detected:
strat_df.loc[strat_df.spot_intensity < 100, 'matrix_class'] = 0
# Apply molecule features stratification only to molecules that were not detected:
strat_df.loc[strat_df.spot_intensity > 100, 'combined'] = 0
# Finally, merge classes:
strat_df['stratification_class'] = strat_df[["combined", "matrix_class"]].applymap(str).apply(''.join, axis=1).astype(
    "category")
strat_df['stratification_class'].value_counts()

intensities['stratification_class'] = strat_df['stratification_class']

# ## Train regression models
# 

# In[35]:


# Define cross-validation objects:
NUM_SPLITS = 10
skf = sklearn.model_selection.StratifiedKFold(n_splits=NUM_SPLITS)
skf.get_n_splits()

# Define training functions with multi outputs:

# In[161]:

from tqdm import tqdm

# from tqdm import notebook

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor


def train_test_regression_models_multi_output(train_x, test_x, train_y, test_y,
                                              out_multi_index=None,
                                              model_set=(),
                                              name_test=None, train=True,
                                              test_multioutout_models=True,
                                              y_is_multioutput=True
                                              ):
    if y_is_multioutput: assert out_multi_index is not None

    results_df = pd.DataFrame()
    regressors = {
        # 'Lin_reg': LinearRegression(),
        # 'Lin_regMultiOut': LinearRegression(),
        # 'SVR_rbf': SVR(kernel='rbf', C=100, gamma='auto'),
        # 'SVR_lin': SVR(kernel='linear', C=100, gamma='auto'), # This works terribly
        # 'SVR_poly': SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1),
        # 'KNeighbors': KNeighborsRegressor(n_neighbors=5),
        # 'DecisionTree': DecisionTreeRegressor(max_depth=5),
        # 'DecisionTreeMultiOut': DecisionTreeRegressor(max_depth=5),
        'RandomForest': RandomForestRegressor(max_depth=5, n_estimators=10),
        # 'RandomForestMultiOut': RandomForestRegressor(max_depth=5, n_estimators=10),
        # 'MLP': MLPRegressor(max_iter=1000),
        # 'MLPMultiOut': MLPRegressor(max_iter=1000),
        # 'GaussianProcess': GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel()),
        # 'GaussianProcessMultiOut': GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel())
    }

    if len(model_set) == 0: model_set = regressors.keys()

    if name_test is not None:
        nb_outputs = test_y.shape[1]
        name_test = name_test.loc[name_test.index.repeat(nb_outputs)]

    pbar = tqdm(model_set, leave=False)
    # for r in model_set:
    for r in pbar:
        pbar.set_postfix({'regressor': r})
        regressor = regressors[r]
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
                loc_res_df['regressor'] = r
                loc_res_df.reset_index(drop=True, inplace=True)
                if name_test is not None:
                    loc_res_df = loc_res_df.merge(name_test.reset_index(drop=True), left_index=True, right_index=True)
            else:
                loc_res_df = pd.DataFrame({'observed_value': test_y,
                                           'prediction': y_pred,
                                           'regressor': r})
                if name_test is not None:
                    loc_res_df = loc_res_df.merge(name_test, left_index=True, right_index=True)
        else:
            raise NotImplementedError()
        results_df = pd.concat([results_df, loc_res_df])

    results_df = results_df.reset_index(drop=True)
    return results_df


# In[162]:

def train_one_regressor_per_matrix(features_normalized, intensity_column="spot_intensity",
                                   train_loop_function=train_test_regression_models_multi_output,
                                   test_split_col_name='combined',
                                   oversampler=None,
                                   ):
    # Cross-validation loop:
    regression_results = pd.DataFrame()

    pbar_matrices = tqdm(intensities.groupby(by=["matrix", "polarity"]), leave=False)
    for (matrix, polarity), rows in pbar_matrices:
        rows = rows.reset_index(drop=True)
        pbar_cross_split = tqdm(skf.split(rows.index, rows[test_split_col_name]),
                                leave=False, total=NUM_SPLITS)

        for fold, (train_index, test_index) in enumerate(pbar_cross_split):
            # pbar_cross_split.set_postfix({'Cross-validation split': r})

            train_intensities = rows.loc[train_index]
            test_intensities = rows.loc[test_index]

            train_y = train_intensities[intensity_column].to_numpy()
            test_y = test_intensities[intensity_column].to_numpy()

            # print(train_intensities[adducts_columns])

            train_x = \
                pd.merge(train_intensities[adducts_columns.tolist() + ["name_short"]],
                         features_normalized,
                         how="left",
                         right_index=True,
                         left_on="name_short"
                         ).drop(columns=["name_short"]).to_numpy()
            test_x = \
                pd.merge(test_intensities[adducts_columns.tolist() + ["name_short", "adduct"]],
                         features_normalized,
                         how="left",
                         right_index=True,
                         left_on="name_short"
                         )

            test_mol_names = test_x[["name_short", "adduct"]].reset_index(drop=True)
            test_x = test_x.drop(columns=["name_short", "adduct"]).to_numpy()

            if oversampler is not None:
                train_x, train_y = oversampler.fit_resample(train_x, train_y)

            results_df = train_loop_function(
                train_x, test_x, train_y, test_y,
                name_test=test_mol_names,
                test_multioutout_models=False,
                y_is_multioutput=False,
                train=True)

            results_df["matrix"] = matrix
            results_df["polarity"] = polarity
            results_df["fold"] = int(fold)
            regression_results = pd.concat([regression_results, results_df])
    return regression_results


def train_multi_output_regressors(features_normalized, intensity_column="spot_intensity",
                                  test_split_col_name='combined'):
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

    pbar_cross_split = tqdm(skf.split(selected_mols.index, selected_mols[test_split_col_name]), leave=False, total=NUM_SPLITS)

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
        results_df = train_test_regression_models_multi_output(train_x, test_x, train_y, test_y,
                                                               name_test=test_mol_names,
                                                               out_multi_index=out_multi_index,
                                                               train=True)
        results_df["fold"] = int(fold)
        regression_results = pd.concat([regression_results, results_df])
    return regression_results


# ----------------------
# Classification
# ----------------------


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


def train_classification_models_multi_output(train_x, test_x, train_y, test_y,
                                             out_multi_index=None,
                                             model_set=(),
                                             name_test=None, train=True,
                                             test_multioutout_models=True,
                                             y_is_multioutput=True):
    if y_is_multioutput: assert out_multi_index is not None
    results_df = pd.DataFrame()
    regressors = {
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

    if len(model_set) == 0: model_set = regressors.keys()

    if name_test is not None and y_is_multioutput:
        nb_outputs = test_y.shape[1]
        name_test = name_test.loc[name_test.index.repeat(nb_outputs)]

    pbar = tqdm(model_set, leave=False)
    # for r in model_set:
    for r in pbar:
        pbar.set_postfix({'regressor': r})
        regressor = regressors[r]
        # These classifiers need to train several models (one per class):
        if "MultiOut" not in r and y_is_multioutput:
            regressor = MultiOutputClassifier(regressor)
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
                loc_res_df['regressor'] = r
                loc_res_df.reset_index(drop=True, inplace=True)
                if name_test is not None:
                    loc_res_df = loc_res_df.merge(name_test.reset_index(drop=True), left_index=True, right_index=True)
            else:
                loc_res_df = pd.DataFrame({'observed_value': test_y,
                                           'prediction': y_pred,
                                           'regressor': r})
                if name_test is not None:
                    loc_res_df = loc_res_df.merge(name_test, left_index=True, right_index=True)

        else:
            raise NotImplementedError()
        results_df = pd.concat([results_df, loc_res_df])

    results_df = results_df.reset_index(drop=True)
    return results_df


# def train_one_det_classifier_per_matrix(features_normalized, intensity_column="spot_intensity"):
#     # Cross-validation loop:
#     regression_results = pd.DataFrame()
#     pbar_cross_split = tqdm(skf.split(digitized_mol_properties.index, digitized_mol_properties['combined']),
#                             leave=False, total=NUM_SPLITS)
#     # tqdm()
#     for train_index, test_index in pbar_cross_split:
#         # pbar_cross_split.set_postfix({'Cross-validation split': r})
#         pbar_matrices = tqdm(intensities.groupby(by=["matrix", "polarity"]), leave=False)
#         for (matrix, polarity), rows in pbar_matrices:
#             train_intensities = rows[rows.name_short.isin(digitized_mol_properties.index[train_index])]
#             test_intensities = rows[rows.name_short.isin(digitized_mol_properties.index[test_index])]
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
#         # Now covert targets to binary labels:
#         test_y = (test_y > 100).astype("int")
#         train_y = (train_y > 100).astype("int")
#
#         test_mol_names = pd.DataFrame([[name, adduct] for (name, adduct), _ in g_test],
#                                       columns=["name_short", "adduct"])
#
#         matrix_names = train_intensities[["matrix", "polarity"]].drop_duplicates()
#         out_multi_index = pd.MultiIndex.from_arrays([matrix_names["matrix"], matrix_names["polarity"]])
#         results_df = train_classification_models_multi_output(train_x, test_x, train_y, test_y,
#                                                               name_test=test_mol_names,
#                                                               out_multi_index=out_multi_index,
#                                                               train=True)
#         results_df["fold"] = int(fold)
#         regression_results = pd.concat([regression_results, results_df])
#     return regression_results


# def train_multioutput_det_classifier(features_normalized, intensity_column="spot_intensity"):
#     # Cross-validation loop:
#     regression_results = pd.DataFrame()
#     selected_mols = digitized_mol_properties
#
#     pbar_cross_split = tqdm(skf.split(selected_mols.index, selected_mols['combined']), leave=False, total=NUM_SPLITS)
#
#     sorted_intensities = intensities.sort_values(by=['name_short', 'adduct', "matrix", "polarity"])
#     sorted_intensities = pd.merge(sorted_intensities,
#                                   features_normalized,
#                                   how="left",
#                                   right_index=True,
#                                   left_on="name_short"
#                                   )
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
#         # Now covert targets to binary labels:
#         test_y = (test_y > 100).astype("int")
#         train_y = (train_y > 100).astype("int")
#
#         test_mol_names = pd.DataFrame([[name, adduct] for (name, adduct), _ in g_test],
#                                       columns=["name_short", "adduct"])
#
#         matrix_names = train_intensities[["matrix", "polarity"]].drop_duplicates()
#         out_multi_index = pd.MultiIndex.from_arrays([matrix_names["matrix"], matrix_names["polarity"]])
#         results_df = train_classification_models_multi_output(train_x, test_x, train_y, test_y,
#                                                               name_test=test_mol_names,
#                                                               out_multi_index=out_multi_index,
#                                                               train=True)
#         results_df["fold"] = int(fold)
#         regression_results = pd.concat([regression_results, results_df])
#     return regression_results


def train_matrix_classifiers(features_normalized, intensity_column="spot_intensity"):
    # Cross-validation loop:
    classification_results = pd.DataFrame()
    selected_mols = digitized_mol_properties

    pbar_cross_split = tqdm(skf.split(selected_mols.index, selected_mols['combined']), leave=False, total=NUM_SPLITS)

    sorted_intensities = intensities.sort_values(by=['name_short', 'adduct', "matrix", "polarity"])
    sorted_intensities = pd.merge(sorted_intensities,
                                  features_normalized,
                                  how="left",
                                  right_index=True,
                                  left_on="name_short"
                                  )

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

        def get_classes(intensities):
            # Now get class labels:
            nb_classes = intensities.shape[1]
            intensities = (intensities > 100).astype("int")
            # Check if all are zero, and then add an additional class:
            not_detected_mask = (intensities.max(axis=1) == 0)
            matrix_class = intensities.argmax(axis=1)
            matrix_class[not_detected_mask] = nb_classes
            return matrix_class

        test_y = get_classes(test_y)
        train_y = get_classes(train_y)

        test_mol_names = pd.DataFrame([[name, adduct] for (name, adduct), _ in g_test],
                                      columns=["name_short", "adduct"])

        matrix_names = train_intensities[["matrix", "polarity"]].drop_duplicates()
        # out_multi_index = pd.MultiIndex.from_arrays([matrix_names["matrix"], matrix_names["polarity"]])
        results_df = train_classification_models_multi_output(train_x, test_x, train_y, test_y,
                                                              name_test=test_mol_names,
                                                              y_is_multioutput=False,
                                                              test_multioutout_models=False,
                                                              train=True)
        results_df["fold"] = int(fold)
        classification_results = pd.concat([classification_results, results_df])
    return classification_results, matrix_names


# Now, train regressors using:
# - Only fingerprints
# - Only mol features
# - Both mol features and fingerprints

# In[163]:


# All features:
import time

task_type = "detection"

if task_type == "regression":
    tick = time.time()
    print("One single mol feature")
    regr_out = result_dir / "regression/random-forest"
    regr_out.mkdir(exist_ok=True, parents=True)
    regression_results_fingerprints = train_multi_output_regressors(features_norm_df[mol_properties_cols[[0]]],
                                                                    intensity_column="norm_intensity_seurat")
    regression_results_fingerprints.to_csv(regr_out / "regr_results_single_mol_feat.csv")
    print('Took {} s'.format(time.time() - tick))

    tick = time.time()
    print("Only FINGERPRINTS")
    regression_results_fingerprints = train_multi_output_regressors(features_norm_df[fingerprints_cols],
                                                                    intensity_column="norm_intensity_seurat")
    regression_results_fingerprints.to_csv(regr_out / "regr_results_fingerprints.csv")
    print('Took {} s'.format(time.time() - tick))

    tick = time.time()
    print("Only MOLECULES")
    regression_results_mols = train_multi_output_regressors(features_norm_df[mol_properties_cols],
                                                            intensity_column="norm_intensity_seurat")
    regression_results_mols.to_csv(regr_out / "regr_results_mol_feat.csv")
    print('Took {} s'.format(time.time() - tick))

    tick = time.time()
    print("Both features")
    regression_results_all_feat = train_multi_output_regressors(features_norm_df,
                                                                intensity_column="norm_intensity_seurat")
    regression_results_all_feat.to_csv(regr_out / "regr_results_all_feat.csv")
    print('Took {} s'.format(time.time() - tick))
elif task_type == "detection":
    # Discretize the intensity:
    intensities["detected"] = (intensities["spot_intensity"] > 100).astype("int")

    # Get oversampler:
    sampler = RandomOverSampler(sampling_strategy="not majority", random_state=43)

    tick = time.time()
    print("Both features")
    regression_results_all_feat = \
        train_one_regressor_per_matrix(features_norm_df,
                                       intensity_column="detected",
                                       train_loop_function=train_classification_models_multi_output,
                                       test_split_col_name="stratification_class",
                                       oversampler=sampler
                                       )
    regression_results_all_feat.to_csv(result_dir / "detection_results_all_feat.csv")
    print('Took {} s'.format(time.time() - tick))
elif task_type == "matrix_classification":
    tick = time.time()
    print("Both features")
    classification_results, matrix_names = train_matrix_classifiers(features_norm_df)
    classification_results.to_csv(result_dir / "matrix_classification_results_all_feat.csv")
    matrix_names.to_csv(result_dir / "matrix_names.csv")
    print('Took {} s'.format(time.time() - tick))

# In[181]:


# import pandas as pd
# import numpy as np
# from tqdm import tqdm
#
# df = pd.DataFrame(np.random.randint(0, 100, (1000000, 100)))
#
# tqdm.pandas(desc="power DataFrame 1M to 100 random int!")
#
# df.progress_apply(lambda x: x**2)
# df.groupby(0).progress_apply(lambda x: x**2)
#
#
# # In[47]:
#
#
# # Save results:
# regression_results.to_csv(plots_dir / "../results/regr_results_fingerprints_plus_feat.csv")
#
#
# # In[49]:
#
#
# # regression_results = pd.read_csv("/Users/alberto-mac/EMBL_repos/spotting-project-regression/results/regr_results_fingerprints_plus_feat.csv", index_col=0)
# # regression_results
#
#
# # ### Evaluate results
#
# # In[50]:
#
#
# import scipy.stats
# from sklearn.metrics import mean_squared_error
#
# # compute Spearman's correlation and mean squared error for
# regression_metrics = pd.DataFrame(columns = ['matrix', 'polarity', 'regressor', "Spearman's R", 'pval', 'RMSE'])
# counter = 0
# for (matrix, polarity, regressor), rows in regression_results.groupby(['matrix', 'polarity', 'regressor']):
#     spearman = scipy.stats.spearmanr(rows.observed_value, rows.prediction)
#     mse = mean_squared_error(rows.observed_value, rows.prediction, squared = False)
#     regression_metrics.loc[counter] = [matrix, polarity, regressor, spearman[0], spearman[1], mse]
#     # print(matrix, polarity, regressor)
#     counter += 1
#
#
# # In[48]:
#
#
# # select best regressor for each matrix/polarity combination
# best_RMSE = regression_metrics.loc[regression_metrics.groupby(['matrix', 'polarity'])["RMSE"].idxmin()]
# # best_RMSE
#
# best_spear = regression_metrics.loc[regression_metrics.groupby(['matrix', 'polarity'])["Spearman's R"].idxmax()].sort_values("Spearman's R", ascending=False)
# best_spear
#
#
# # Legacy: remove molecules that are never detected from results (from now on, those are not included in training)
#
# # In[91]:
#
#
# # Temp fix to get rid of molecules that were never detected from results dt:
# g = regression_results.groupby(['name_short', 'adduct'], as_index=False)["observed_value"].max()
# regression_results = pd.merge(regression_results, g[g["observed_value"] != 0.][["adduct", "name_short"]],
#         how="inner")
#
#
# # Check if the best matrix is selected for each ion with a selected regressor
#
# # In[180]:
#
#
# accuracy_df = pd.DataFrame(columns = ['regressor', 'accuracy'])
# for i, selected_regressor in enumerate(regression_results["regressor"].unique()):
#     accuracy = 0
#     for (molecule, adduct), rows in regression_results[regression_results.regressor == selected_regressor].groupby(['name_short', 'adduct']):
#         best_observed = rows.loc[rows["observed_value"].idxmax()][["matrix", "polarity"]]
#         best_predicted = rows.loc[rows["prediction"].idxmax()][["matrix", "polarity"]]
#         assert len(best_observed.shape) == 1, 'Same intensity observed for multiple combinations of matrices and polarities: {}'.format(rows["observed_value"])
#
#         if len(best_predicted.shape) == 1:
#             if (best_observed == best_predicted).all(): accuracy += 1
#         else:
#             print("Warning: multiple predictions with same intensity")
#             # Multiple matrices with same predicted score. Check if the best is one of them:
#             if len(pd.merge(best_observed, best_predicted)) > 0: accuracy += 1
#
#     accuracy = accuracy / regression_results[regression_results.regressor == selected_regressor][['name_short', 'adduct']].drop_duplicates().shape[0]
#     accuracy_df.loc[i] = [selected_regressor, accuracy]
#
#
# # In[178]:
#
#
# accuracy_df.sort_values("accuracy", ascending=False)
#
#
# # In[ ]:
#
#
#
#
