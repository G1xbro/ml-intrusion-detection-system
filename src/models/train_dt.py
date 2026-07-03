import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_model():
    print("🏋️‍♂️ Loading processed arrays from disk...")
    # Load the compressed data we saved in Phase 2
    X_train = np.load("./data/processed/X_train.npy")
    X_test = np.load("./data/processed/X_test.npy")
    y_train = np.load("./data/processed/y_train.npy")
    y_test = np.load("./data/processed/y_test.npy")
    
    print(f"📊 Training Data Size: {X_train.shape[0]} rows")
    print(f"📊 Testing Data Size: {X_test.shape[0]} rows")
    
    # 1. Initialize the Model
    # Setting max_depth prevents the tree from over-complexifying and freezing your PC
    print("\n🤖 Initializing Baseline Decision Tree Classifier...")
    model = DecisionTreeClassifier(max_depth=12, random_state=42)
    
    # 2. Train the Model
    print("⏳ Training model on network logs (This may take a minute)...")
    model.fit(X_train, y_train)
    print("✨ Model training complete!")
    
    # 3. Evaluate the Model
    print("\n🔮 Making predictions on the unseen test set...")
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n=================== 📊 PERFORMANCE REPORT ===================")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Metrics (Focus on Recall/F1-Score for Attacks!):")
    # target_names maps 0 to Benign and 1 to Attack
    print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=============================================================")
    
    # 4. Save the trained model artifact
    os.makedirs("./models/", exist_ok=True)
    model_path = "./models/baseline_dt_model.joblib"
    print(f"\n💾 Exporting trained model to {model_path}...")
    joblib.dump(model, model_path)
    print("✅ Training pipeline finished successfully!")

if __name__ == "__main__":
    train_model()