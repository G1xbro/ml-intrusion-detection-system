import pandas as pd

# Load our scoped dataset
df = pd.read_csv("./data/oscp_scope_traffic.csv")

# Fix the encoding character bug so we can filter accurately
df['Label'] = df['Label'].str.replace(r'\s*\ufffd\s*', ' - ', regex=True)

# Separate based on the text 'BENIGN' vs anything else (Attacks)
attacks = df[df['Label'] != 'BENIGN'].sample(15, random_state=42)
benign = df[df['Label'] == 'BENIGN'].sample(5, random_state=42)

# Combine and shuffle
test_stream = pd.concat([attacks, benign]).sample(frac=1, random_state=7)

# Save the live simulation sample file
test_stream.to_csv("./data/sample_live_traffic.csv", index=False)
print("✅ Fixed! Created ./data/sample_live_traffic.csv with 20 mixed packets for simulation.")