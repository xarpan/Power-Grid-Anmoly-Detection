import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load data set
df = pd.read_csv('Data/binary_supervised/powersys_data_supervised_binary.csv')  



#Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# Separate features and labels (if available)
X_train = df.drop(columns=['marker'], errors='ignore')  # Features

# Define the model with a radial basis function (RBF) kernel
svm = OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')  # nu is the outlier fraction

# Train the model
svm.fit(X_train)
print("Model trained successfully!")

# Predict anomalies
test_features = X_test.drop(columns=['marker'], errors='ignore')  # Features
test_labels = X_test['marker'] if 'marker' in df.columns else None  # Labels (optional)

# Anomaly predictions (1 = normal, -1 = anomaly)
test_predictions = svm.predict(test_features)
test_predictions = np.where(test_predictions == 1, 1, 0)  # Convert to 0 (normal) and 1 (anomaly)

print("Anomaly detection completed!")

# Calculate evaluation metrics
print("Classification Report:")
print(classification_report(test_labels, test_predictions))

accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions)  # Anomalies are now labeled as 1
recall = recall_score(test_labels, test_predictions)
f1 = f1_score(test_labels, test_predictions)

# Display the results
print(f"Accuracy: {accuracy:.4f}") 
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
