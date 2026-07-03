import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_xgb():
    print("🏋️‍♂️ Loading processed arrays from disk...")
    X_train = np.load("./data/processed/X_train.npy")
    X_test = np.load("./data/processed/X_test.npy")
    y_train = np.load("./data/processed/y_train.npy")
    y_test = np.load("./data/processed/y_test.npy")
    
    print("\n🚀 Initializing XGBoost Classifier...")
    # max_depth=6 is lightweight but highly effective for boosting
    # n_jobs=-1 ensures all your CPU cores handle the parallel calculations
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("⏳ Training XGBoost (This might take a moment, boosting runs sequentially)...")
    model.fit(X_train, y_train)
    print("✨ Model training complete!")
    
    print("\n🔮 Making predictions...")
    y_pred = model.predict(X_test)
    
    print("\n=================== 📊 XGB PERFORMANCE REPORT ===================")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=============================================================")
    
    os.makedirs("./models/", exist_ok=True)
    joblib.dump(model, "./models/xgb_model.joblib")
    print("✅ XGBoost model saved successfully!")

if __name__ == "__main__":
    train_xgb()