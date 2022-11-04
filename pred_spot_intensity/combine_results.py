from pathlib import Path

import pandas as pd


def combine_results_from_multiple_experiments(all_exp_names):
    """
    Combine all results of classification and regression in single pandas dataframe (returned as output).
    If multiple iterations of the same experiment were taken, then it also generates a collected csv file.
    """
    assert isinstance(all_exp_names, (list, tuple))
    assert len(all_exp_names) > 0
    collected = {}

    for exp_path, all_files_to_process in all_exp_names:
        exp_path = Path(exp_path)
        assert exp_path.is_dir(), exp_path

        if "detection" in str(exp_path):
            metric = "macro_avg_f1_score"
            count_key = 'detected'
            task_typename = "classification"
            model_typename = "classifier"
        elif "regression" in str(exp_path) or "rank" in str(exp_path):
            metric = "Spearman's R"
            count_key = 'non-zero obs'
            task_typename = "regression"
            model_typename = "regressor"
        else:
            raise ValueError(exp_path)

        for file_to_process in all_files_to_process:
            collected_df = []
            iter = 0
            while True:
                df_path = exp_path / (file_to_process + f"_{iter}.csv")
                if not df_path.is_file():
                    break
                loc_df = pd.read_csv(df_path, index_col=0)
                # loc_df.loc[:, "regressor"] = loc_df["regressor"] + f"_{iter}"
                # loc_df.loc[:, "classifier"] = loc_df["classifier"] + f"_{iter}"
                loc_df["iter_index"] = iter
                collected_df.append(loc_df)
                iter += 1
            assert len(collected_df) != 0
            collected_df = pd.concat(collected_df).reset_index(drop=True)
            collected_df.to_csv(exp_path / (file_to_process + ".csv"))

            # Now aggregate results for Veronika...?
            if "random" not in file_to_process:
                exported_predictions = collected_df.drop(columns=[model_typename, "fold"])
                index_columns = ["matrix", "polarity", "name_short"]

                if task_typename == "regression":
                    mean_predictions = exported_predictions.groupby(index_columns, as_index=True).mean().drop(
                        columns="iter_index")
                elif task_typename == "classification":
                    total_nb_iterations = exported_predictions.iter_index.max() + 1
                    mean_predictions = exported_predictions.groupby(index_columns, as_index=True).sum().drop(
                        columns="iter_index")
                    mean_predictions /= float(total_nb_iterations)
                    mean_predictions = (mean_predictions > 0.5).astype("int")
                else:
                    raise ValueError(task_typename)
                collected[task_typename] = mean_predictions

    # Reformat collected predictions:
    out_df = collected["regression"].join(collected["classification"].drop(columns=["observed_value"]), lsuffix="_regression",
                                 rsuffix="_classification").reset_index()

    out_df["prediction_classification"] = out_df["prediction_classification"] == 1
    out_df.loc[~out_df["prediction_classification"], "prediction_regression"] = ''
    out_df = out_df.rename(columns={"name_short": "Molecule name",
                         "observed_value": "Measured intensity (log10[intensity+1])",
                         "prediction_regression": "Predicted intensity (log10[intensity+1])",
                         "prediction_classification": "Predicted as detected"})
    return out_df

