# Spotting Project: Predict matrix intensities using a Machine Learning model
The trained models and predictions can be found in the [training_results/paper_results](./training_results/paper_results) folder.

### How to Install
- `conda create --name spottingProjIntensityEnv python=3.8`
- `conda activate spottingProjIntensityEnv`
- `pip install -r requirements.txt`
- `python setup.py install`

### Predicting matrix intensities for custom molecules
Follow the instructions in the notebook [predict_intensities.ipynb](./predict_intensities.ipynb) to predict matrix intensities on your custom set of molecules.

### Reproducing paper results
To reproduce the paper results by following these steps:
- Retrain the models by running `python train_models.py --exp_name <YOUR_NEW_EXPERIMENT_NAME>`, where `YOUR_NEW_EXPERIMENT_NAME` could be `reproduce_paper_results`
- After training is done, models and predictions can be found in the `training_results/YOUR_NEW_EXPERIMENT_NAME` folder
- To compute scores and make plots, use the [evaluate_trained_models.ipynb](./evaluate_trained_models.ipynb) notebook

