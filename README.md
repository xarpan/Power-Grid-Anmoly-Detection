# ⚡ Power System Anomaly Detection App

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)  
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-TensorFlow%2C%20RandomForest%2C%20SupportVectorMachine%2C%20Autoencoder-orange)  
![Streamlit](https://img.shields.io/badge/Streamlit+-success)  



## 🔥 Overview

The **Power System Anomaly Detection App** is a machine learning-powered dashboard designed to detect and visualize anomalies such as cyberattacks on smart power systems in real time. This tool provides operators with critical insights to ensure system reliability, prevent failures and improve response time to stop cyber security attacks on the power system.

---

## 🚀 Features
- **📊 Interactive Dashboard**:
  
  The app provides a user-friendly dashboard built stream to visualize key features of an electrical power system, notify users of detected anomalies, and display event log data of anomalies detected.
  - Uses a trained Random Forest binary classification model to detect anomalies.  
- **Simulated Real-Time Data Stream**:
  - Generates data dynamically based on historical system data for testing and demonstrations.
- **Dynamic Visualization**:
  - Live graphs showing critical features such as bus voltage  
- **Event Log**:
  - Log display of detected anomalies with timestamps and affected feature values. Log data can be downloaded from the dashboard in .csv format for further analaysis. 
- **🌟 Future Enhancements**:
  - Deployment and integration with power system operators SCADA system/IoT for actual system monitoring and anomaly detection.
  - Multiclass model training to predict the type of anomally to operator
  - Add predictive maintenance capabilities.
  - AExtend visualization for additional metrics (e.g., power factor, harmonic distortion).

---

## 🛠️ Technologies

This app is built with the following technologies:

- **Backend + Frontend**: Streamlit
- **Machine Learning**: TensorFlow, XGBoost, RandomForest, IsolationForest, Autoencoder
- **Data Visualization**: Matplotlib

---

## 📂 Repository Structure

├── ML training + Data        # ML training scripts and data files

│   ├── Data                  # Raw and processed labelled data files


├── webapp                    # Files for local web app deployment

│   ├── app.py                # Python app for power system anomally detection


├── README.md                 # Project documentation

---

## 🚀 Getting Started

### Prerequisites:
1. Python 3.12 or later
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

### Running the App locally;
1. Start the Streamlit server: Run below code from the app directory in terminal
    ```bash
    streamlit run app.py

2. Access the app: App opens dashboard in default browser automatically, and URL is displayed in the terminal as well.

---

## 🧑‍💻 Machine Learning Development Workflow
### Data sources and processing:
- The data source for this project is from the [Power System Attack Datasets by the Mississippi State University and Oak Ridge National Laboratory - 4/15/2014](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets) .It has three data set for anomally classification: binary, three-class, and multi-class. The data set I used in the implementation model was the binary data set(Normal operation vs Attack Events).
- This dataset is simulated and includes status flags and power system parameters monitored by 4 Intelligent Electronic Devices that can switch circuit breakers on or off automatically or operated manually. An overview of the power system and monitoring and control devices are as seen below:
  
![image](https://github.com/user-attachments/assets/93f62811-387e-4a38-89af-f742c667eb39)
- Comprehensive documentation of the data set can be found [here](http://www.google.com/url?q=http%3A%2F%2Fwww.ece.uah.edu%2F~thm0009%2Ficsdatasets%2FPowerSystem_Dataset_README.pdf&sa=D&sntz=1&usg=AOvVaw3t-soxdA-27GPUSRG1JP_Q) .
- The cleaned labelled binary classification dataset has the follow properties:
  - 128 features of which 128 were used in training the ML models
  - 72074 rows of data  
### ML Model selection and training:
3 supervised learning ML Models and 3 unsupervised learning ML were trained with a training data to test data split of 80% and 20%. Model performance was assessed with the following results:



## 📄 License
This project is licensed under the MIT License.

## 💡 Acknowledgments
- Scikit-learn
- TensorFlow
- Matplotlib
