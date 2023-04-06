# Predicting AP-MALDI response using Machine Learning
The trained models and predictions can be found in the [training_results/paper_results](./training_results/paper_results) folder.

### How to Install
- `conda create --name apMALDIresponseEnv python=3.8`
- `conda activate apMALDIresponseEnv`
- `pip install -r requirements.txt`
- `python setup.py install`

### Predicting AP-MALDI response for custom molecules
Follow the instructions in the notebook [predict_intensities.ipynb](./predict_intensities.ipynb) to predict AP-MALDI response on your custom set of molecules.

### Reproducing paper results
To reproduce the paper results, follow these steps:
- Retrain the models by running `python train_models.py --exp_name <YOUR_NEW_EXPERIMENT_NAME>`
- After training is done, models and predictions can be found in the `training_results/YOUR_NEW_EXPERIMENT_NAME` folder
- To compute scores and make plots, use the [evaluate_trained_models.ipynb](./evaluate_trained_models.ipynb) notebook

