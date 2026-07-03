import pandas as pd
import numpy as np
import glob
import os

# 1. Define paths
DATA_DIR = "../data/"
OUTPUT_FILE = os.path.join(DATA_DIR, "oscp_scope_traffic.csv")

# 2. Select files matching your OSCP scope to keep the dataset size manageable
# We include Monday for baseline normal behavior, plus PortScan, BruteForce, and Web Attacks.
target_files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
]

all_dfs = []

print("📂 Starting data combination pipeline...")

for file_name in target_files:
    file_path = os.path.join(DATA_DIR, file_name)
    
    if os.path.exists(file_path):
        print(f"📖 Reading: {file_name}...")
        # Read the file
        df = pd.read_csv(file_path)
        
        # FIX 1: Strip whitespaces from column headers on the fly
        df.columns = df.columns.str.strip()
        
        all_dfs.append(df)
    else:
        print(f"❌ Warning: Missing expected file {file_name}")

# 3. Combine selected days into one master DataFrame
print("\n🔄 Merging datasets...")
combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)

# FIX 2: Handle the "Infinity" and "NaN" traps
print("🧹 Cleaning infinite values and missing rows...")
# Replace infinite values with NumPy's NaN object
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop any row that contains a NaN value
combined_df.dropna(inplace=True)

# 4. Take a look at your class distributions (Your OSCP targets!)
print("\n📊 Target Class Distribution across selected scope:")
print(combined_df['Label'].value_counts())

# 5. Export to a single optimized CSV file
print(f"\n💾 Saving processed dataset to {OUTPUT_FILE}...")
combined_df.to_csv(OUTPUT_FILE, index=False)
print("✅ Pipeline complete! Ready for Exploratory Data Analysis (EDA).")