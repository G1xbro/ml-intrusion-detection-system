import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def run_preprocessing():
    print("🚀 Starting Preprocessing Pipeline...")
    
    # Paths
    input_path = "./data/oscp_scope_traffic.csv"
    output_dir = "./data/processed/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    print("📖 Loading dataset...")
    df = pd.read_csv(input_path)
    
    # 2. Fix character encoding issue in labels (Your awesome regex fix!)
    df['Label'] = df['Label'].str.replace(r'\s*\ufffd\s*', ' - ', regex=True)
    
    # 3. Create Binary Target (0 = Benign, 1 = Attack)
    print("🎯 Engineering target variables...")
    df['Binary_Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # 4. Define columns to drop immediately
    cols_to_drop = [
        'Label', 'Binary_Label', # Separate targets from features
        'Fwd Header Length.1',   # Duplicate column
        # Zero-variance/Constant columns
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
        'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
        # Highly redundant subflow features
        'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes'
    ]
    
    # Separate Features (X) and Target (y)
    X = df.drop(columns=cols_to_drop, errors='ignore')
    y = df['Binary_Label']
    
    print(f"📉 Reduced feature count from 79 down to {X.shape[1]}")
    
    # 5. Train/Test Split (80% train, 20% test)
    # Using stratify ensures both sets get an equal proportion of attacks
    print("⚖️ Splitting data into Train and Test sets (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 6. Feature Scaling
    # Crucial for models like Logistic Regression or Neural Networks so large numbers don't dominate
    print("⚖️ Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Save the processed numpy arrays and the scaler object
    print("💾 Saving processed datasets and scaler object...")
    np.save(os.path.join(output_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train.to_numpy())
    np.save(os.path.join(output_dir, "y_test.npy"), y_test.to_numpy())
    
    # Save the scaler so we can use it during live IDS simulation later
    os.makedirs("./models/", exist_ok=True)
    joblib.dump(scaler, "./models/scaler.joblib")
    
    print("✅ Preprocessing pipeline complete! Clean arrays saved in /data/processed/")

if __name__ == "__main__":
    run_preprocessing()