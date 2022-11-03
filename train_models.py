import os
import subprocess
import sys

from pred_spot_intensity.combine_results import combine_results_from_multiple_experiments

# TODO:
#   - notebook to predict with model
#   - clean notebook to create plots
#   - clean inputs / results / plots
#   - clean package code


if __name__ == "__main__":
    EXP_NAME = "reproduce_article_results"

    python_interpreter = sys.executable
    # TODO:
    #  - update iters number
    #  - reactivate command to run model training
    all_options_to_run = [
        # f"-s mol random --task_type regression_on_detected_per_mol --nb_iter 2 --experiment_name {EXP_NAME}",
        # f"-s mol random --task_type detection_per_mol --nb_iter 2 --experiment_name {EXP_NAME}",
        f"-s mol --task_type regression_on_detected_per_mol --do_feat_sel --only_save_feat --experiment_name {EXP_NAME}",
        f"-s mol --task_type detection_per_mol --do_feat_sel --only_save_feat --experiment_name {EXP_NAME}",
    ]
    for i, options in enumerate(all_options_to_run):
        full_command = '"{}" -m pred_spot_intensity {} {}'.format(
            python_interpreter,
            "--" if "ipython" in python_interpreter else "",
            options
        )
        print(f"\n\n\n####### Running set of experiments {i + 1}/{len(all_options_to_run)}... ###### ")
        print(full_command)
        subprocess.run(full_command, shell=True, check=True)

    # TODO: merge results from multiple iterations using the other script
    all_exp_names = [
        (f"./results/{EXP_NAME}/regression_on_detected_per_mol_sum", ["results_mol_feat", "results_random_feat"]),
        (f"./results/{EXP_NAME}/detection_per_mol_sum", ["results_mol_feat", "results_random_feat"]),
    ]

    all_results = combine_results_from_multiple_experiments(all_exp_names)
    all_results.to_csv(f"./results/{EXP_NAME}/all_paper_results.csv")
