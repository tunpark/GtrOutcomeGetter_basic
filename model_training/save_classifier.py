import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import json
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

class LogisticSMOTEModelSaver:
    """
    Trains and saves a specific model configuration: Logistic Regression with SMOTE.
    
    This class encapsulates the final training pipeline, evaluates the chosen model
    on a hold-out set, and saves all necessary artifacts (model, vectorizer, metadata)
    for future predictions.
    """
    
    def __init__(self, file_path, best_weight=0.05, model_dir="final_model"):
        self.file_path = file_path
        self.best_weight = best_weight
        self.model_dir = model_dir
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.smote = None
        self.metrics = {}
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print(f"Created model directory: {self.model_dir}")
    
    def load_and_prepare_data(self):
        """Loads and preprocesses the data from the source file."""
        print("="*50)
        print("Loading and preparing data...")
        print("="*50)
        
        df = pd.read_csv(self.file_path)
        df = df.dropna(subset=["text", "label", "source"])
        df["label"] = df["label"].astype(int)
        df["text"] = df["text"].apply(lambda x: " ".join(str(x).split()[:256]))
        
        self.df = df
        print(f"Loaded {len(df)} valid records.")
        class_dist = df['label'].value_counts().sort_index()
        print(f"Class distribution: {dict(class_dist)}")
    
    def train_final_model(self):
        """Trains the final Logistic Regression + SMOTE model and evaluates it."""
        print("\n" + "="*50)
        print("Training Final Logistic Regression + SMOTE Model")
        print("="*50)
        
        X = self.df["text"]
        y = self.df["label"]
        X_vec = self.vectorizer.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y, test_size=0.2, stratify=y, random_state=42
        )
        
        print(f"Training set distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Test set distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
        # Apply SMOTE to the training data
        print("\nApplying SMOTE oversampling...")
        k_neighbors = min(5, max(1, np.sum(y_train == 1) - 1))
        self.smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train, y_train)
        print(f"Distribution after SMOTE: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")

        # Train the model
        print("\nTraining Logistic Regression model...")
        self.model.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate the model on the untouched test set
        print("\nEvaluating model on the test set:")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Store metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        self.metrics = {
            'accuracy': report['accuracy'],
            'f1_score': report['weighted avg']['f1-score'],
            'auc': roc_auc_score(y_test, y_proba),
            'minority_f1': report.get('1', {}).get('f1-score', 0),
            'minority_precision': report.get('1', {}).get('precision', 0),
            'minority_recall': report.get('1', {}).get('recall', 0)
        }
        
        print(f"Overall F1 Score: {self.metrics['f1_score']:.4f}")
        print(f"AUC Score: {self.metrics['auc']:.4f}")
        print(f"Minority Class F1 Score: {self.metrics['minority_f1']:.4f}")
        print("\nFull Classification Report:")
        print(classification_report(y_test, y_pred))

    def save_model_artifacts(self):
        """Saves the trained model, vectorizer, and metadata to disk."""
        print("\n" + "="*50)
        print("Saving model and artifacts...")
        print("="*50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model, vectorizer, and smote objects
        model_path = os.path.join(self.model_dir, f"model_{timestamp}.joblib")
        joblib.dump(self.model, model_path)
        vectorizer_path = os.path.join(self.model_dir, f"vectorizer_{timestamp}.joblib")
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save metadata
        metadata = {
            'model_name': 'Logistic Regression with SMOTE',
            'timestamp': timestamp,
            'metrics': self.metrics,
            'model_path': model_path,
            'vectorizer_path': vectorizer_path
        }
        metadata_path = os.path.join(self.model_dir, f"metadata_{timestamp}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Create a simple config file to point to the latest artifacts
        config = {'model_file': os.path.basename(model_path), 'vectorizer_file': os.path.basename(vectorizer_path)}
        config_path = os.path.join(self.model_dir, "latest_model_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Configuration file updated: {config_path}")
    
    def run_and_save(self):
        """Executes the full pipeline: load, train, and save."""
        try:
            self.load_and_prepare_data()
            self.train_final_model()
            self.save_model_artifacts()
            print("\nModel training and saving process completed successfully.")
        except Exception as e:
            print(f"\nAn error occurred during the process: {e}")
            import traceback
            traceback.print_exc()

def load_model_and_predict(texts, model_dir="final_model"):
    """Loads the latest model from a directory and makes predictions."""
    config_path = os.path.join(model_dir, "latest_model_config.json")
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Load artifacts
    model = joblib.load(os.path.join(model_dir, config['model_file']))
    vectorizer = joblib.load(os.path.join(model_dir, config['vectorizer_file']))
    
    # Predict
    if isinstance(texts, str): texts = [texts]
    X_vec = vectorizer.transform(texts)
    predictions = model.predict(X_vec)
    probabilities = model.predict_proba(X_vec)
    
    print("\nPrediction Results:")
    print("-" * 50)
    for text, pred, prob in zip(texts, predictions, probabilities):
        print(f"Text: \"{text[:80]}...\"")
        print(f"  -> Prediction: {pred} (Confidence for class 1: {prob[1]:.2%})")
        print("-" * 50)
        
    return predictions, probabilities

if __name__ == "__main__":
    DATA_FILE = 'combined_analysis_data.csv'
    
    # --- Main Workflow ---
    # 1. Train and save the model
    if os.path.exists(DATA_FILE):
        print("Found data file. Starting training and saving process...")
        trainer = LogisticSMOTEModelSaver(DATA_FILE)
        trainer.run_and_save()
        
        # 2. Example of loading and predicting with the saved model
        print("\n--- Testing Prediction with Saved Model ---")
        test_texts = [
            "This is a new software package we developed for data analysis.", # Should be 1
            "Our study resulted in a new creative methodology for artistic expression." # Should be 0
        ]
        load_model_and_predict(test_texts)
    else:
        print(f"\nData file '{DATA_FILE}' not found.")
        print("Please run the previous analysis scripts to generate it.")