import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Load model and test data
model = joblib.load('../models/random_forest_model.pkl')
data = pd.read_csv('../data/preprocessed_creditcard.csv')
X = data.drop(columns=['Class'])
y = data['Class']

# Prediction and evaluation
y_pred = model.predict(X)
print("Classification Report:\n", classification_report(y, y_pred))
print("ROC AUC Score:", roc_auc_score(y, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
