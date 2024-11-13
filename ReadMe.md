# Credit Card Fraud Detection Project

## Overview
This project applies machine learning techniques to detect fraudulent transactions in credit card data. It showcases skills in feature engineering, handling imbalanced data, and evaluating classification models.

## Dataset
The dataset comes from [Kaggle's Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), containing transactions labeled as fraud or non-fraud.

## Requirements
- Python 3.x
- Jupyter Notebook
- Required libraries: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, xgboost

## Project Structure
- `data/`: Contains the dataset.
- `notebooks/`: Jupyter notebooks for each stage.
- `scripts/`: Python scripts for data preprocessing, model training, and evaluation.
- `models/`: Folder to save trained models.

## Steps to Run
1. Install the requirements.
2. Run each notebook in order: `EDA`, `Feature Engineering`, `Model Training`.
3. To retrain using scripts, use `scripts/data_preprocessing.py`, `scripts/model_training.py`, and `scripts/model_evaluation.py`.
