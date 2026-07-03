# Machine Learning Intrusion Detection System (ML-IDS)

An intelligent Intrusion Detection System built using modern Machine Learning pipelines to detect network anomalies and malicious vectors. This project uses the **CICIDS2017** benchmark network dataset to focus heavily on reconnaissance and initial access footprints relevant to **OSCP** study targets.

## 📊 Project Features & Scope
This project scopes down millions of rows of raw corporate network logs into key target attack vectors:
* **Port Scanning:** Detecting rapid sequential sweeps across network ports (e.g., Nmap).
* **Credential Brute Forcing:** Identifying authentication hammering signatures (FTP/SSH Patator attacks).
* **Web Exploitation:** Flagging Cross-Site Scripting (XSS) and SQL Injection payload behaviors based on forward/backward stream anomalies.

## 📂 Repository Structure
```text
├── data/                       # Local dataset storage
|   ├── oscp_scope_traffic.csv
│   └── sample_live_traffic.csv
├── models/                     # Trained serialization binaries (.joblib)
│   ├── scaler.joblib
│   └── xgb_model.joblib
├── notebooks/                  # Exploratory Data Analysis & Visualization
│   └── 01_eda_and_visualization.ipynb
├── src/                        # Production-ready codebase
│   ├── models/
│   │   ├── train_dt.py
│   │   ├── train_lr.py
│   │   ├── train_rf.py
│   │   └── train_xgb.py
│   └── preprocess.py
├── app.py                      # Interactive Network Simulation Dashboard
└── generate_test_stream.py     # Data stream emulator utility
```

## 📈 Model Performance Matrix
Evaluated multiple foundational and ensemble classification algorithms to track overall `accuracy`, `precision`, and minimizing dangerous `False Negatives` (missed exploits):

|Model Architecture|Accuracy|False Positives|False Negatives|Status|
|------------------|--------|---------------|---------------|------|
|Logistic Regression|99.04%|1,377|1,380|Baseline|
|Decision Tree|99.98%|11|42|Robust|
|Random Forest|99.99%|0|29|Excellent|
|XGBoost|99.99%|3|17|Production Winner|

---

## 💻 How to Run the Simulation
1. Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/ml-intrusion-detection-system.git](https://github.com/YOUR_GITHUB_USERNAME/ml-intrusion-detection-system.git)
    ```
    ```bash
    cd ml-intrusion-detection-system
    ```
2. Set up virtual environment & dependencies:
    ```bash
    python3 -m venv venv
    ```
    ```bash
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```
    ```
    pip install -r requirements.txt
    ```
3. Execute the Engine Simulator:
    ```bash
    python app.py
    ```