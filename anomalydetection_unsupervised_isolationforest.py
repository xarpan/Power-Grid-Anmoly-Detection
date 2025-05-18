import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load data set
df = pd.read_csv('Data/binary_supervised/powersys_data_supervised_binary.csv')  

#Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# Separate features and labels (if available)
X_train = df.drop(columns=['marker'], errors='ignore')  # Features



# Train Isolation Forest
isolation_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
isolation_forest.fit(X_train)

print("Model trained successfully!")

# Predict anomalies
test_features = X_test.drop(columns=['marker'], errors='ignore')  # Features
test_labels = X_test['marker'] if 'marker' in df.columns else None # Labels 

# Anomaly predictions (1 = normal, -1 = anomaly)
test_predictions = isolation_forest.predict(test_features)
test_predictions = [1 if x == -1 else 0 for x in test_predictions]  # Convert to 0 (normal) and 1 (anomaly)

print("Anomaly detection completed!")

if test_labels is not None:
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
else:
    print("No true labels available for evaluation.")