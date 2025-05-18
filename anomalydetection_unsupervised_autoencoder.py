import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Load and preprocess the dataset
df = pd.read_csv('Data/binary_supervised/powersys_data_supervised_binary.csv')  

y_test = df['marker'] if 'marker' in df.columns else None # Labels 
train = df.drop(columns=['marker'], errors='ignore')  # Features

scaler = StandardScaler()
X = scaler.fit_transform(train)


# Define the autoencoder
input_dim = X.shape[1]
encoding_dim = 16

autoencoder = Sequential([
    Dense(encoding_dim, activation='relu', input_dim=input_dim),
    Dense(8, activation='relu'),
    Dense(encoding_dim, activation='relu'),
    Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(
    X, X,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=1
)

# Calculate reconstruction error
reconstructed = autoencoder.predict(X)
reconstruction_error = np.mean(np.square(X - reconstructed), axis=1)

# Set threshold and classify anomalies
threshold = np.percentile(reconstruction_error, 95)
y_pred = (reconstruction_error > threshold).astype(int)

# Example ground truth (if available)

print("Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)  # Anomalies are now labeled as 1
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy:.4f}") 
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")