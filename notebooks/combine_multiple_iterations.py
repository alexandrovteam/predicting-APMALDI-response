import os
from pathlib import Path
import pandas as pd
result_folder = Path("/Users/alberto-mac/EMBL_repos/spotting-project-regression/results")

all_exp_names = [
    ("regression_on_detected_per_mol_sum/sklearn_v1", ["results_mol_feat", "results_random_feat"]),
    ("detection_per_mol_sum/sklearn_v1", ["results_mol_feat", "results_random_feat"]),
    # OLD:
    # ("regression_on_detected/skorch_v3_hidden_size_32", ["results_mol_feat", "results_random_feat"]),
    # ("regression_on_detected/skleanr_MLP_v1", ["results_mol_feat", "results_random_feat"]),
    # ("regression_on_detected/skorch_v3_hidden_size_100", ["results_mol_feat"])
]

collected = {}

for exp_name, all_files_to_process in all_exp_names:
    input_path = result_folder / exp_name
    assert input_path.is_dir(), input_path

    if "detection" in exp_name:
        metric = "macro_avg_f1_score"
        count_key = 'detected'
        task_typename = "classification"
        model_typename = "classifier"
    elif "regression" in exp_name or "rank" in exp_name:
        metric = "Spearman's R"
        count_key = 'non-zero obs'
        task_typename = "regression"
        model_typename = "regressor"
    else:
        raise ValueError(exp_name)

    for file_to_process in all_files_to_process:
        collected_df = []
        iter = 0
        while True:
            df_path = input_path / (file_to_process + f"_{iter}.csv")
            if not df_path.is_file():
                print(df_path)
                break
            loc_df = pd.read_csv(df_path, index_col=0)
            # loc_df.loc[:, "regressor"] = loc_df["regressor"] + f"_{iter}"
            # loc_df.loc[:, "classifier"] = loc_df["classifier"] + f"_{iter}"
            loc_df["iter_index"] = iter
            collected_df.append(loc_df)
            iter += 1
        assert len(collected_df) != 0
        collected_df = pd.concat(collected_df).reset_index(drop=True)
        collected_df.to_csv(input_path / (file_to_process + ".csv"))

        # Now aggregate results for Veronika...?
        if "random" not in file_to_process:
            exported_predictions = collected_df.drop(columns=[model_typename, "fold"])
            index_columns = ["matrix", "polarity", "name_short"]

            if task_typename == "regression":
                mean_predictions = exported_predictions.groupby(index_columns, as_index=True).mean().drop(columns="iter_index")
            elif task_typename == "classification":
                total_nb_iterations = exported_predictions.iter_index.max() + 1
                mean_predictions = exported_predictions.groupby(index_columns, as_index=True).sum().drop(columns="iter_index")
                mean_predictions /= float(total_nb_iterations)
                mean_predictions = (mean_predictions > 0.5).astype("int")
            else:
                raise ValueError(task_typename)
            collected[task_typename] = mean_predictions

collected["regression"].join(collected["classification"].drop(columns=["observed_value"]), lsuffix="_regression", rsuffix="_classification").reset_index()\
    .rename(columns={"name_short": "Molecule name"})\
    .to_csv(input_path / ("combined_avg_predictions.csv"))

# mean_predictions.reset_index().to_csv(input_path / (file_to_process + "_avg_predictions.csv"))
