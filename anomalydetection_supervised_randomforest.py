import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('Data/multiclass_supervised/powersys_data_supervised_triple.csv')  # Replace with your dataset path


# Separate features and labels
X = df.drop(columns=['marker'])  # Features
scaler = StandardScaler()         # Normalize features
X = scaler.fit_transform(X)

y = df['marker']                # Labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model training completed!")

y_pred = model.predict(X_test)

print("Predictions completed!")

# Confusion matrix and classification report
#print("Confusion Matrix:")
#print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#joblib.dump(model, 'supervised_anomaly_model.pkl')
#print("Model saved as 'supervised_anomaly_model.pkl'")
