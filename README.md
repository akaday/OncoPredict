# OncoPredict

## Introduction
OncoPredict is a project aimed at detecting cancer from blood samples using machine learning and deep learning techniques.

## Project Structure
- `data/`: Datasets used for training and testing.
- `notebooks/`: Jupyter notebooks for experiments and data exploration.
- `src/`: Source code for model training and evaluation.
- `models/`: Trained models and their configurations.
- `results/`: Results and performance metrics.
- `tests/`: Test scripts for validating data and models.

## How to Run
1. Clone the repo: `git clone https://github.com/your-username/OncoPredict.git`
2. Install Python dependencies: `pip install tensorflow pandas`
3. Install npm dependencies: `npm install`
4. Run the training script: `python src/train.py`

## Hyperparameter Tuning
To run hyperparameter tuning using Optuna, follow these steps:
1. Install the required dependencies: `pip install optuna`
2. Run the training script with hyperparameter tuning: `python src/train.py --tune`

## Generating and Viewing Evaluation Metrics and Visualizations
To generate and view detailed evaluation metrics and visualizations, follow these steps:
1. Ensure you have the required dependencies: `pip install scikit-learn matplotlib`
2. Run the training script: `python src/train.py`
3. After training, evaluation metrics will be saved in the `results/metrics.txt` file.
4. Confusion matrix and ROC curve visualizations will be saved in the `results/` directory as `confusion_matrix.png` and `roc_curve.png` respectively.
