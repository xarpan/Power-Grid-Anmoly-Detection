import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Title for the app
st.title("Real-Time Power System Anomaly Detection Dashboard")
# Short description
st.markdown(
    """
    This dashboard monitors simulates real-time power system data , detects anomalies, and visualizes key metrics. 
    It applies machine learning models to identify potential faults or irregularities, ensuring reliable system performance.
    """
)
st.markdown(
    """
    ---
    Developed by: Nana Yaw Owusu Ofori-Ampofo | [LinkedIn](https://linkedin.com/in/nyoofori-ampofo) | [GitHub](https://github.com/NanaYawOA)
    """
)





# Load the scaler, model, and feature names
scaler = joblib.load('scaler.pkl')
binary_rf_model = joblib.load('supervised_anomaly_model_binary.pkl')
feature_names = joblib.load('feature_names.pkl')  # Ensure feature alignment

# Load the original dataset to simulate data properties
df = pd.read_csv(
    'datasets/binary/powersys_data_supervised_binary_sample space.csv'
)  


# Get statistical properties for simulation
mean = df.mean()
std = df.std()

# Identify binary and discrete columns

binary_columns = df.select_dtypes(include=['int8', 'bool']).columns.tolist()
discrete_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()



# Custom function for inference validation
def preprocess_for_inference(batch):
    """
    Validate incoming data, align features, and preprocess for inference.
    """
    # Validate feature alignment
    missing_features = [col for col in feature_names if col not in batch.columns]
    extra_features = [col for col in batch.columns if col not in feature_names]

    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    if extra_features:
        st.warning(f"Ignoring extra features: {extra_features}")

    # Align and preprocess
    batch_aligned = batch[feature_names]
    batch_scaled = scaler.transform(batch_aligned)
    return batch_scaled

# Function to generate simulated real-time data
def generate_real_time_data(num_samples=1, anomaly_prob=0.1, variability_scale=0.1):
    data = []
    for _ in range(num_samples):
        discrete_sample = np.random.normal(mean[discrete_columns], std[discrete_columns] * variability_scale)
        binary_sample = np.random.choice([0, 1], size=len(binary_columns), p=[0.7, 0.3])
        combined_sample = np.concatenate([discrete_sample, binary_sample])

        if np.random.rand() < anomaly_prob:
            # Add anomaly to random discrete feature
            if len(discrete_columns) > 0:  # Ensure discrete_columns is not empty 
                anomaly_feature = np.random.choice(discrete_columns)
                anomaly_index = df.columns.get_loc(anomaly_feature)
                combined_sample[anomaly_index] += np.random.uniform(-5 * std[anomaly_feature], 5 * std[anomaly_feature])

            # Flip a random binary feature
            if len(binary_columns) > 0 and np.random.rand() < 0.1:  # Ensure binary_columns is not empty
                binary_anomaly_feature = np.random.choice(binary_columns)
                binary_anomaly_index = df.columns.get_loc(binary_anomaly_feature)
                combined_sample[binary_anomaly_index] = 1 - combined_sample[binary_anomaly_index]

        data.append(combined_sample)
    return pd.DataFrame(data, columns=df.columns)

# Stream simulated data
def stream_data(batch_size=1, total_batches=100):
    for _ in range(total_batches):
        yield generate_real_time_data(num_samples=batch_size)
        time.sleep(1)

# Placeholders for Streamlit
real_time_placeholder = st.empty()
anomaly_placeholder = st.empty()
graph_placeholder = st.empty()
event_log_placeholder = st.empty()

# Event log
event_log = []
graph_data = pd.DataFrame(columns=df.columns)

# Streaming real-time data
for batch in stream_data(batch_size=1, total_batches=100):
    # Handle missing or invalid data
    batch.fillna(0, inplace=True)

    # Process data for inference
    try:
        batch_scaled = preprocess_for_inference(batch)
    except ValueError as e:
        st.error(f"Inference Error: {e}")
        continue

    # Make predictions
    predictions = binary_rf_model.predict(batch_scaled)

    # Check for anomalies
    anomalies_detected = np.any(predictions == 1)
    if anomalies_detected:
        anomaly_placeholder.error("Anomaly Detected!")
        for i, pred in enumerate(predictions):
            if pred == 1:
                event_log.append({"Timestamp": pd.Timestamp.now(), "Data": batch.iloc[i].to_dict()})
    else:
        anomaly_placeholder.success("Normal operation")

    # Append batch to graph data
    graph_data = pd.concat([graph_data, batch], ignore_index=True)
    if len(graph_data) > 100:  # Keep the last 100 rows for visualization
        graph_data = graph_data.iloc[-50:]

    # Visualize real-time data
    with graph_placeholder.container():
        col1, col2 = st.columns([2, 1])  # 3:1 column ratio for graph and image

    # Graph on the left
    with col1:
        r1_v = graph_data[['R1-PM1:V', 'R1-PM2:V', 'R1-PM3:V']].mean(axis=1)
        r2_v = graph_data[['R2-PM1:V', 'R2-PM2:V', 'R2-PM3:V']].mean(axis=1)
        r3_v = graph_data[['R3-PM1:V', 'R3-PM2:V', 'R3-PM3:V']].mean(axis=1)
        r4_v = graph_data[['R4-PM1:V', 'R4-PM2:V', 'R4-PM3:V']].mean(axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(r1_v, label="BR1 Bus Voltage", color='blue')
        ax.plot(r2_v, label="BR2 Bus Voltage", color='red')
        ax.plot(r3_v, label="BR3 Bus Voltage", color='black')
        ax.plot(r4_v, label="BR4 Bus Voltage", color='yellow')
        ax.set_title("Real-Time Bus Voltage Monitoring")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Voltage (V)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # Image on the right
    with col2:
        st.image("assets/system.jpg", use_container_width=True)
    # Display event log
    if event_log:
        event_log_placeholder.write(pd.DataFrame(event_log).tail(10))

    time.sleep(1)  # Simulate real-time streaming


