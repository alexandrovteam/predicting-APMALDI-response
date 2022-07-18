from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
import os

from pred_spot_intensity.sklearn_training_utils import train_one_model_per_matrix_polarity, get_strat_classes
from pred_spot_intensity.train_pytorch_models import train_pytorch_model_on_intensities

plt.style.use('dark_background')


def train_models(args):
    TASK_TYPE = args.task_type
    DO_FEAT_SEL = args.do_feat_sel
    ONLY_SAVE_FEAT = args.only_save_feat
    NUM_SPLITS = args.nb_splits

    setups = args.setup_list

    PRED_VAL_THRESH = 0.2
    ION_AGGREGATE_RULE = args.ion_aggregate_rul

    # ----------------------------
    # LOAD AND NORMALIZE DATA:
    # ----------------------------
    # Paths:
    current_dir_path = Path(os.path.dirname(os.path.realpath(__file__)))

    input_dir = current_dir_path / "../input_data"
    plots_dir = current_dir_path / "../plots"
    plots_dir.mkdir(exist_ok=True)
    result_dir = current_dir_path / "../results"
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
    intensities = pd.read_csv(input_dir / "intensity_data_raw.csv", index_col=0)
    intensities = intensities.rename(columns={"Matrix short": "matrix", "Polarity": "polarity",
                                              "spot_intensity_bgr_corrected": "spot_intensity",
                                              "detectability": "detected"})

    # 14583 removed
    # Remove matrix-obscured entries:
    intensities = intensities[intensities.matrix_obscured == 0]

    # Remove everything that is not trusted according to pred_val:
    intensities = intensities[
        (intensities.pred_val <= PRED_VAL_THRESH) | (intensities.pred_val >= (1. - PRED_VAL_THRESH))]
    # Overwrite detected column according to the
    intensities.loc[:, "detected"] = intensities.pred_val >= (1. - PRED_VAL_THRESH)

    # Set not-detectable intensities to zero:
    intensities.loc[intensities.detected == 0, "spot_intensity"] = 0

    # Sanity checks:
    nb_before = len(intensities.name_short.unique())

    # Delete molecules with missing properties:
    intensities = intensities[~intensities.name_short.isin(missing_molecules)]
    print("{}/{} molecules kept".format(len(intensities.name_short.unique()), nb_before))

    remove_not_detected_adducts = False

    # Convert adducts to one-hot encoding:
    adducts = intensities[['adduct']].drop_duplicates()
    adducts["adduct_name"] = adducts.adduct
    adducts = adducts.set_index("adduct", drop=True)
    adducts_one_hot = pd.get_dummies(adducts.adduct_name, prefix='adduct')
    adducts_columns = adducts_one_hot.columns

    # TODO: avoid merging with intensity df
    intensities = intensities.merge(adducts_one_hot, how="right", on="adduct")

    # ## Methods for standartization/normalization
    # First, normalize features

    ss = StandardScaler()
    pt = PowerTransformer()

    # OPTION 1
    mol_properties_norm_df = pd.DataFrame(pt.fit_transform(mol_properties),
                                          index=mol_properties.index,
                                          columns=mol_properties.columns)

    features_norm_df = pd.merge(mol_properties_norm_df, fingerprints, how="inner", right_index=True, left_index=True)

    intensities["norm_intensity"] = np.log10(intensities['spot_intensity'] + 1)

    # # Get max intensities across adducts:
    if ION_AGGREGATE_RULE == "max":
        aggregated_intesities_per_mol = intensities.groupby(["name_short", "matrix", "polarity"], as_index=False)[
            ["norm_intensity", "spot_intensity", "detected"]].max()
    elif ION_AGGREGATE_RULE == "sum":
        aggregated_intesities_per_mol = intensities.groupby(["name_short", "matrix", "polarity"], as_index=False)[
            ["spot_intensity"]].sum()
        aggregated_intesities_per_mol["detected"] = intensities.groupby(["name_short", "matrix", "polarity"],
                                                                        as_index=False)[
            ["detected"]].max()["detected"]
        aggregated_intesities_per_mol["norm_intensity"] = np.log10(aggregated_intesities_per_mol['spot_intensity'] + 1)
    else:
        raise ValueError(ION_AGGREGATE_RULE)

    # ----------------------------
    # CREATE TRAIN/VAL SPLIT:
    # ----------------------------

    # Since not all the bins have enough datapoints, use quantiles to define the size of the bins:

    # We only select only some features, otherwise there are not enough data in each of the splits:
    selected_stratification_features = [
        "pka_strongest_basic",
        # "polar_surface_area",
        # "polarizability"
    ]
    # selected_stratification_features = mol_properties_cols

    digitized_mol_properties = pd.DataFrame(index=features_norm_df.index)
    for col in selected_stratification_features:
        digitized_mol_properties[col] = pd.qcut(features_norm_df[col], q=2, labels=[1, 2])

    digitized_mol_properties['mol_strat_class'] = digitized_mol_properties.astype(str).sum(axis=1).astype('category')

    if "detection" in TASK_TYPE:
        aggregated_intesities_per_mol['stratification_class'] = get_strat_classes(aggregated_intesities_per_mol,
                                                                                  digitized_mol_properties,
                                                                                  "detected",
                                                                                  stratify_not_detected=False)
        intensities['stratification_class'] = get_strat_classes(intensities,
                                                                digitized_mol_properties,
                                                                "detected",
                                                                stratify_not_detected=False)
    elif TASK_TYPE == "intensity_classification":
        aggregated_intesities_per_mol['stratification_class'] = get_strat_classes(aggregated_intesities_per_mol,
                                                                                  digitized_mol_properties,
                                                                                  "digitized_seurat")
        intensities['stratification_class'] = get_strat_classes(intensities,
                                                                digitized_mol_properties,
                                                                "digitized_seurat")
    elif "regression" in TASK_TYPE:
        intensities['stratification_class'] = intensities.merge(digitized_mol_properties,
                                                                left_on="name_short",
                                                                right_index=True,
                                                                how="left")["mol_strat_class"]
        aggregated_intesities_per_mol['stratification_class'] = aggregated_intesities_per_mol.merge(digitized_mol_properties,
                                                                left_on="name_short",
                                                                right_index=True,
                                                                how="left")["mol_strat_class"]

    elif TASK_TYPE == "regression_on_all":
        raise DeprecationWarning()
        # intensities['stratification_class'] = get_strat_classes(intensities, digitized_mol_properties,
        #                                                         "detected")

    # ----------------------------
    # START TRAINING:
    # ----------------------------

    # Now, train regressors using:
    # - Only fingerprints
    # - Only mol features
    # - Both mol features and fingerprints

    # All features:
    import time

    FEATURES_TYPE = None

    dir_out = result_dir / TASK_TYPE
    if "per_mol" in TASK_TYPE:
        dir_out = result_dir / f"{TASK_TYPE}_{ION_AGGREGATE_RULE}"
    dir_out.mkdir(exist_ok=True, parents=True)

    random_features = pd.DataFrame(np.random.normal(size=features_norm_df.shape[0]), index=features_norm_df.index)
    zero_features = pd.DataFrame(np.zeros(shape=features_norm_df.shape[0], dtype="float"), index=features_norm_df.index)

    runs_setup = {
        "fingerprints": [features_norm_df[fingerprints_cols]],
        "mol": [features_norm_df[mol_properties_cols]],
        "all": [features_norm_df],
        "random": [random_features],
        "no": [zero_features],
    }
    FEAT_SEL_CSV_FILE = None

    for setup_name in setups:
        assert setup_name in runs_setup
        if DO_FEAT_SEL:
            assert args.nb_iter == 1, "Multiple iterations not supported for feature selection"
            if setup_name == "fingerprints":
                FEATURES_TYPE = "categorical"
            elif setup_name == "mol":
                FEATURES_TYPE = "numerical"
            else:
                raise ValueError(f"{setup_name} not supported for feature selection")

        for iter in range(args.nb_iter):
            if not DO_FEAT_SEL:
                if args.nb_iter == 1:
                    out_filename = f"results_{setup_name}_feat.csv"
                else:
                    out_filename = f"results_{setup_name}_feat_{iter}.csv"
            else:
                out_filename = f"{setup_name}_{'feature_importance' if ONLY_SAVE_FEAT else 'feat_selection_results'}.csv"

            if args.feat_sel_load_dir is not None:
                FEAT_SEL_CSV_FILE = result_dir / args.feat_sel_load_dir / f"{setup_name}_feature_importance.csv"

            # Start running:
            tick = time.time()
            print(f"Running setup {setup_name}...")
            if "regression" in TASK_TYPE:
                assert TASK_TYPE == "regression_on_detected" or TASK_TYPE == "regression_on_detected_per_mol"
                use_adduct_features = "per_mol" not in TASK_TYPE
                model_results = \
                    train_one_model_per_matrix_polarity(intensities if use_adduct_features else aggregated_intesities_per_mol,
                                                        runs_setup[setup_name][0],
                                                        intensity_column="norm_intensity",
                                                        type_of_models="regressor",
                                                        test_split_col_name="stratification_class",
                                                        use_adduct_features=use_adduct_features,
                                                        train_only_on_detected="on_detected" in TASK_TYPE,
                                                        adducts_columns=adducts_columns,
                                                        do_feature_selection=DO_FEAT_SEL,
                                                        only_save_feat_sel_results=ONLY_SAVE_FEAT,
                                                        features_type=FEATURES_TYPE,
                                                        path_feature_importance_csv=FEAT_SEL_CSV_FILE,
                                                        num_cross_val_folds=NUM_SPLITS
                                                        )

            elif "detection" in TASK_TYPE:
                assert TASK_TYPE == "detection_per_mol" or TASK_TYPE == "detection_per_ion"
                # Discretize the intensity:
                # Get oversampler:
                sampler = RandomOverSampler(sampling_strategy="not majority", random_state=43)
                use_adduct_features = TASK_TYPE == "detection_per_ion"
                model_results = \
                    train_one_model_per_matrix_polarity(
                        intensities if use_adduct_features else aggregated_intesities_per_mol,
                        runs_setup[setup_name][0],
                        intensity_column="detected",
                        type_of_models="classifier",
                        test_split_col_name="stratification_class",
                        use_adduct_features=use_adduct_features,
                        oversampler=sampler, adducts_columns=adducts_columns,
                        do_feature_selection=DO_FEAT_SEL,
                        only_save_feat_sel_results=ONLY_SAVE_FEAT,
                        features_type=FEATURES_TYPE,
                        path_feature_importance_csv=FEAT_SEL_CSV_FILE,
                        num_cross_val_folds=NUM_SPLITS
                        )
            elif TASK_TYPE == "rank_matrices" or TASK_TYPE == "pytorch_nn_detect":
                # TODO: rename TASK_TYPE and name...
                task_names = {
                    "rank_matrices": "ranking",
                    "pytorch_nn_detect": "detection"
                }
                if TASK_TYPE == "rank_matrices":
                    task_name = "ranking"
                model_results = train_pytorch_model_on_intensities(intensities,
                                                                   runs_setup[setup_name][0],
                                                                   adducts_one_hot,
                                                                   task_names[TASK_TYPE],
                                                                   do_feature_selection=DO_FEAT_SEL,
                                                                   path_feature_importance_csv=FEAT_SEL_CSV_FILE,
                                                                   num_cross_val_folds=2  # TODO: update
                                                                   )

            else:
                raise ValueError(f"Task type not recognized {TASK_TYPE}")

            print('Took {} s'.format(time.time() - tick))

            # Write results:
            model_results.to_csv(dir_out / out_filename)
