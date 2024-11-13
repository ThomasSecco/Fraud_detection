import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load preprocessed data
data = pd.read_csv('../data/preprocessed_creditcard.csv')
X = data.drop(columns=['Class'])
y = data['Class']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, '../models/random_forest_model.pkl')
