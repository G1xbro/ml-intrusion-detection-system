import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_lr():
    print("🏋️‍♂️ Loading processed arrays from disk...")
    X_train = np.load("./data/processed/X_train.npy")
    X_test = np.load("./data/processed/X_test.npy")
    y_train = np.load("./data/processed/y_train.npy")
    y_test = np.load("./data/processed/y_test.npy")
    
    print("\n📈 Initializing Logistic Regression Classifier...")
    # max_iter ensures the solver has enough steps to find the pattern
    model = LogisticRegression(max_iter=500, random_state=42)
    
    print("⏳ Training Logistic Regression...")
    model.fit(X_train, y_train)
    print("✨ Model training complete!")
    
    print("\n🔮 Making predictions...")
    y_pred = model.predict(X_test)
    
    print("\n=================== 📊 LR PERFORMANCE REPORT ===================")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=============================================================")
    
    os.makedirs("./models/", exist_ok=True)
    joblib.dump(model, "./models/lr_model.joblib")
    print("✅ LR model saved successfully!")

if __name__ == "__main__":
    train_lr()