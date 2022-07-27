import argparse
import os
from pathlib import Path

# FIXME: add allRank repo to the path
import sys
current_dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(os.path.normpath(current_dir_path / "../../allRank")))

from pred_spot_intensity.train_models import train_models


if __name__ == "__main__":
    # For the moment it only supports model training
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default="regression_on_all")
    # parser.add_argument('--feature_selection', type=str, default=None)
    parser.add_argument('--do_feat_sel', action='store_true')
    parser.add_argument('--only_save_feat', action='store_true')
    parser.add_argument('--nb_iter', type=int, default=1)
    parser.add_argument('--nb_splits', type=int, default=10)
    parser.add_argument('--feat_sel_load_dir', type=str, default=None)
    parser.add_argument('--postfix', type=str, default=None)
    parser.add_argument('--pred_val_thresh', type=float, default=0.2)
    parser.add_argument('--ion_aggregate_rul', type=str, default="max")
    parser.add_argument('-s', '--setup_list', nargs='+', default=[])
    args = parser.parse_args()
    train_models(args)
