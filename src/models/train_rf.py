import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_rf():
    print("🏋️‍♂️ Loading processed arrays from disk...")
    X_train = np.load("./data/processed/X_train.npy")
    X_test = np.load("./data/processed/X_test.npy")
    y_train = np.load("./data/processed/y_train.npy")
    y_test = np.load("./data/processed/y_test.npy")
    
    print("\n🌲 Initializing Random Forest Classifier...")
    # n_estimators=50 uses 50 trees. n_jobs=-1 forces your PC to use ALL CPU cores to speed it up!
    model = RandomForestClassifier(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42)
    
    print("⏳ Training Random Forest (Using all CPU cores)...")
    model.fit(X_train, y_train)
    print("✨ Model training complete!")
    
    print("\n🔮 Making predictions...")
    y_pred = model.predict(X_test)
    
    print("\n=================== 📊 RF PERFORMANCE REPORT ===================")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=============================================================")
    
    os.makedirs("./models/", exist_ok=True)
    joblib.dump(model, "./models/rf_model.joblib")
    print("✅ RF model saved successfully!")

if __name__ == "__main__":
    train_rf()