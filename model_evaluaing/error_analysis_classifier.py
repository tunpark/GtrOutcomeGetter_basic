import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
    roc_curve, precision_recall_curve
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class SavedModelErrorAnalyzer:
    """
    Performs a detailed error analysis on a saved text classifier.
    
    Workflow:
    1. Loads a pre-trained model and its corresponding vectorizer.
    2. Reproduces the original test set split for consistent evaluation.
    3. Makes predictions and calculates a comprehensive set of metrics.
    4. Exports detailed reports, including all predictions, False Positives, and False Negatives.
    5. Generates a suite of plots for visual analysis of model performance and errors.
    """

    def __init__(
        self,
        data_path="combined_analysis_data.csv",
        model_path="final_model/logistic_smote_model_20250801_145958.joblib",
        vectorizer_path="final_model/logistic_smote_vectorizer_20250801_145958.joblib",
        output_dir="analysis_results",
        topk_fn=50  # Export the top K most confident false negatives for manual review.
    ):
        self.data_path = data_path
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.output_dir = output_dir
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.topk_fn = topk_fn

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.df = None
        self.model = None
        self.vectorizer = None
        self.test_data = None
        self.X_test_text = None
        self.y_test = None

    def load_model_and_vectorizer(self):
        """Loads the saved scikit-learn model and TF-IDF vectorizer."""
        print(f"Loading trained model and vectorizer:\n  - {self.model_path}\n  - {self.vectorizer_path}")
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        print("Model and vectorizer loaded successfully.")

    def load_and_prepare_data(self):
        """Loads and preprocesses the analysis dataset."""
        print(f"\nReading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError("Dataset must contain 'text' and 'label' columns.")

        df = df.dropna(subset=["text", "label"]).copy()
        df["label"] = df["label"].astype(int)
        df["text"] = df["text"].apply(lambda x: " ".join(str(x).split()[:256]))
        df["source"] = df.get("source", "unknown").astype(str)
        
        cls = df["label"].value_counts().sort_index()
        print(f"Class distribution: {dict(cls)} | Imbalance Ratio ~ {cls.max()/max(1, cls.min()):.1f}:1")
        self.df = df

    def make_test_split(self):
        """Recreates the exact same test split used during training for consistency."""
        X, y, idx = self.df["text"], self.df["label"], np.arange(len(self.df))
        _, X_test, _, y_test, _, idx_test = train_test_split(
            X, y, idx, test_size=0.2, stratify=y, random_state=42
        )
        self.X_test_text = X_test
        self.y_test = y_test
        self.test_data = self.df.iloc[idx_test].reset_index(drop=True)
        print(f"\nRecreated test set: {len(X_test)} samples")
        print(f"  Test set class distribution: {dict(pd.Series(y_test).value_counts().sort_index())}")

    def predict_on_test(self):
        """Makes predictions on the test set using the loaded model."""
        print("\nPredicting on the test set...")
        X_test_vec = self.vectorizer.transform(self.X_test_text)
        y_pred = self.model.predict(X_test_vec)
        
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_test_vec)[:, 1]
        else:
            # Fallback for models without predict_proba (e.g., some linear SVMs)
            y_proba = (y_pred == 1).astype(float)
        return y_pred, y_proba

    def export_results(self, y_pred, y_proba):
        """Exports detailed predictions, False Negatives, and False Positives to CSV files."""
        print("\nExporting prediction details and error samples...")
        results_df = pd.DataFrame({
            "text": self.test_data["text"].values,
            "source": self.test_data["source"].values,
            "true_label": self.y_test.values,
            "predicted_label": y_pred,
            "probability_class_1": y_proba,
        })
        results_df["is_correct"] = (results_df["true_label"] == results_df["predicted_label"])
        results_df["error_category"] = ""  # For manual error annotation

        fn_df = results_df[(results_df["true_label"] == 1) & (results_df["predicted_label"] == 0)].copy()
        fp_df = results_df[(results_df["true_label"] == 0) & (results_df["predicted_label"] == 1)].copy()

        # Define output paths
        all_path = os.path.join(self.output_dir, "test_set_predictions.csv")
        fn_path = os.path.join(self.output_dir, "false_negatives.csv")
        fp_path = os.path.join(self.output_dir, "false_positives.csv")
        fn_top_path = os.path.join(self.output_dir, f"false_negatives_top{self.topk_fn}.csv")

        # Save files
        results_df.to_csv(all_path, index=False, encoding="utf-8-sig")
        fn_df.to_csv(fn_path, index=False, encoding="utf-8-sig")
        fp_df.to_csv(fp_path, index=False, encoding="utf-8-sig")

        if len(fn_df) > 0 and self.topk_fn > 0:
            fn_df.sort_values("probability_class_1", ascending=True).head(self.topk_fn)\
                 .to_csv(fn_top_path, index=False, encoding="utf-8-sig")

        print(f"  All predictions saved to: {all_path} ({len(results_df)} rows)")
        print(f"  False Negatives (FN) saved to: {fn_path} ({len(fn_df)} rows)")
        print(f"  False Positives (FP) saved to: {fp_path} ({len(fp_df)} rows)")

    def print_metrics(self, y_pred, y_proba):
        """Calculates and prints key performance metrics for the test set."""
        print("\nTest Set Metrics:")
        print(classification_report(self.y_test, y_pred, digits=4, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))

        try:
            tmp = pd.DataFrame({"is_correct": (self.y_test.values == y_pred), "source": self.test_data["source"].values})
            src_err = tmp.groupby("source")["is_correct"].apply(lambda x: 1 - x.mean()).sort_values(ascending=False)
            print("\nError Rate by Data Source:")
            for source, rate in src_err.items():
                print(f"  {source:<20} -> {rate:.3f}")
        except Exception:
            pass

    def _savefig(self, fig, filename):
        """Helper to save a matplotlib figure."""
        path = os.path.join(self.plots_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {path}")

    def generate_plots(self, results_df, y_pred, y_proba):
        """Generates and saves a suite of analysis visualizations."""
        print("\nGenerating analysis plots...")
        # Plot 1: Confusion Matrix Heatmap
        cm = confusion_matrix(self.y_test, y_pred)
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')
        self._savefig(fig1, "confusion_matrix.png")

        # Plot 2: ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc = roc_auc_score(self.y_test, y_proba)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
        ax2.plot([0, 1], [0, 1], '--', lw=1, color='gray')
        ax2.set_title('ROC Curve'); ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
        ax2.legend(); ax2.grid(alpha=0.3)
        self._savefig(fig2, "roc_curve.png")

        # Plot 3: Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.plot(recall, precision, lw=2)
        ax3.set_title('Precision-Recall Curve'); ax3.set_xlabel('Recall'); ax3.set_ylabel('Precision')
        ax3.grid(alpha=0.3)
        self._savefig(fig3, "precision_recall_curve.png")
        
        # Plot 4: Text Length Distribution for Errors
        LENGTH_THRESHOLD = 500
        NUM_BINS = 20
        results_df['text_length'] = results_df['text'].str.split().str.len()
        fn_lengths = results_df.loc[results_df['is_correct'] == False, 'text_length'].dropna()
        fn_lengths_capped = fn_lengths.clip(upper=LENGTH_THRESHOLD)
        bins = np.linspace(0, LENGTH_THRESHOLD, NUM_BINS + 1)
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.hist(fn_lengths_capped, bins=bins, alpha=0.7, label='Errors', color='salmon')
        xticks = ax4.get_xticks()
        xticklabels = [f'{int(x)}' for x in xticks]
        if xticks[-1] >= LENGTH_THRESHOLD:
            xticklabels[-1] = f'{LENGTH_THRESHOLD}+'
        ax4.set_xticks(xticks)
        ax4.set_xticklabels(xticklabels)
        ax4.set_title('Text Length Distribution of Errors')
        ax4.set_xlabel('Text Length (Number of Words)'); ax4.set_ylabel('Count')
        ax4.legend(); ax4.grid(alpha=0.3)
        self._savefig(fig4, "text_length_distribution_errors.png")

    def run(self):
        """Executes the full analysis pipeline."""
        self.load_model_and_vectorizer()
        self.load_and_prepare_data()
        self.make_test_split()
        y_pred, y_proba = self.predict_on_test()
        self.print_metrics(y_pred, y_proba)
        results_df = self.export_results(y_pred, y_proba)
        self.generate_plots(results_df, y_pred, y_proba)
        print("\nAnalysis finished successfully.")

if __name__ == "__main__":
    analyzer = SavedModelErrorAnalyzer(
        data_path="combined_analysis_data.csv",
        model_path="final_model/model.joblib",
        vectorizer_path="final_model/vectorizer.joblib",
        output_dir="classification_analysis_results"
    )
    analyzer.run()
