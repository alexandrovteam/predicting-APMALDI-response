import argparse
import subprocess
import sys

from pred_spot_intensity.combine_results import combine_results_from_multiple_experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="reproduce_paper_results")
    args = parser.parse_args()

    EXP_NAME = args.exp_name

    python_interpreter = sys.executable
    all_options_to_run = [
        f"-s mol random --task_type regression_on_detected_per_mol --nb_iter 10 --experiment_name {EXP_NAME}",
        f"-s mol random --task_type detection_per_mol --nb_iter 10 --experiment_name {EXP_NAME}",
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

    all_exp_names = [
        (f"./training_results/{EXP_NAME}/regression_on_detected_per_mol_sum", ["results_mol_feat", "results_random_feat"]),
        (f"./training_results/{EXP_NAME}/detection_per_mol_sum", ["results_mol_feat", "results_random_feat"]),
    ]

    all_results = combine_results_from_multiple_experiments(all_exp_names)
    all_results.to_csv(f"./training_results/{EXP_NAME}/collected_predictions.csv")
