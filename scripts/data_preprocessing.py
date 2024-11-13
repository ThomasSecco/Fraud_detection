import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load Data
data = pd.read_csv('../data/creditcard.csv')

# Feature Scaling
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])

# Splitting features and labels
X = data.drop(columns=['Class', 'Time'])
y = data['Class']

# Balance the data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Save processed data
processed_data = pd.concat([X_res, y_res], axis=1)
processed_data.to_csv('../data/preprocessed_creditcard.csv', index=False)
