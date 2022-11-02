# spotting-project-regression

### TODO list: 
- check the final plots/results
  - Notebook for making plots and scores/results of paper
  - Notebook for predicting stuff for unknown molecules
- Put in readme or create a notebook? At least for using/predicting with the trained models?


### Install
- `conda create --name spottingProjIntensityEnv python=3.8`
- `conda activate spottingProjIntensityEnv`
- `pip install -r requirements.txt`
- `python setup.py install`

### Reproducing paper results
- Add splits and number of iterations!
- `python -m pred_spot_intensity -s mol random --task_type regression_on_detected --pred_val_thresh 0.2 --postfix release_tests`
- `python -m pred_spot_intensity -s mol --task_type regression_on_detected --pred_val_thresh 0.2 --do_feat_sel --only_save_feat --postfix release_tests`

### Stuff to do
- Feature selection plots
- Train command
- How to evaluate model
- Install for training
  - Install with torch...?
  - Skip pytorch and clean it in a new branch
