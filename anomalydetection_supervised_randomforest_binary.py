import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv('Data/binary_supervised/powersys_data_supervised_binary.csv')  

# Separate features and labels
X = df.drop(columns=['marker'])  # Features
y = df['marker']                # Labels

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')
print("Feature names saved as 'feature_names.pkl'")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessing pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Apply pipeline to training and test sets
X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Save the preprocessing pipeline
joblib.dump(pipeline, 'preprocessing_pipeline.pkl')
print("Pipeline saved as 'preprocessing_pipeline.pkl'")

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_transformed, y_train)
print("Model training completed!")

# Save the trained model
joblib.dump(model, 'supervised_anomaly_model_binary.pkl')
print("Model saved as 'supervised_anomaly_model_binary.pkl'")

# Evaluate the model
y_pred = model.predict(X_test_transformed)

print("\nClassification Report: 0 (normal) and 1 (anomaly)")
print(classification_report(y_test, y_pred))

# Accuracy and performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
