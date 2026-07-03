import pandas as pd
import numpy as np
import joblib
import time
import os

def run_ids_simulation():
    print("🛡️  Initializing Machine Learning Intrusion Detection System...")
    
    # 1. Load the model and scaler artifacts
    model_path = "./models/xgb_model.joblib"
    scaler_path = "./models/scaler.joblib"
    sample_data_path = "./data/sample_live_traffic.csv"
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("❌ Error: Trained model or scaler missing! Please run your training scripts first.")
        return
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 2. Read the simulated live data stream
    print("📖 Loading simulated live network interface stream...")
    live_traffic = pd.read_csv(sample_data_path)
    
    # Keep the real ground truth label separate so we can verify if the model is right
    ground_truth = live_traffic['Label'].values
    
    # Define columns to drop exactly like preprocessing phase
    cols_to_drop = [
        'Label', 'Binary_Label', 'Fwd Header Length.1',
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
        'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
        'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes'
    ]
    
    features = live_traffic.drop(columns=cols_to_drop, errors='ignore')
    
    print("\n[+] IDS Engine Active. Monitoring Interface 'eth0' for anomalies...")
    print("-" * 90)
    print(f"{'TIMESTAMP':<12} | {'DEST PORT':<10} | {'FLOW DUR (ms)':<15} | {'IDS DETECTION STATUS':<22} | {'ACTUAL TRAFFIC type'}")
    print("-" * 90)
    
    # 3. Simulate processing packet records one-by-one with a slight delay
    for i in range(len(features)):
        single_row = features.iloc[[i]]
        actual_type = ground_truth[i]
        
        # Get quick display telemetry from the packet
        dest_port = int(single_row['Destination Port'].values[0])
        duration = float(single_row['Flow Duration'].values[0])
        
        # Scale the features using our trained scaler
        scaled_row = scaler.transform(single_row)
        
        # Predict status (0 = Benign, 1 = Attack)
        prediction = model.predict(scaled_row)[0]
        
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        
        if prediction == 1:
            # ALERT MODE (Red/High Visibility indication)
            status = "🚨 ALERT: MALICIOUS"
        else:
            status = "🟢 PASS: BENIGN"
            
        print(f"{timestamp:<12} | {dest_port:<10} | {duration:<15.2f} | {status:<22} | ({actual_type})")
        time.sleep(0.6) # Simulates delay between incoming packet flows
        
    print("-" * 90)
    print("✅ Live stream simulation finished. 0 critical errors encountered.")

if __name__ == "__main__":
    run_ids_simulation()