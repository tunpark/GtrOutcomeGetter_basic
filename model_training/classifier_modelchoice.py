import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class TraditionalTextClassifier:
    """
    Analyzes the effect of weak label weighting on various traditional ML models.

    This class loads a dataset with 'gold' and 'weak' labels, trains multiple
    classifiers (e.g., Logistic Regression, SVM) using different sample weights
    for the weak data, and evaluates performance to find the optimal configuration.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "Linear SVM": LinearSVC()
        }
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.results_by_weight = {}

    def load_and_prepare_data(self):
        """Loads and preprocesses data from the specified file path."""
        print("="*50)
        print("Step 1: Loading and preparing data")
        print("="*50)
        df = pd.read_csv(self.file_path)
        df = df.dropna(subset=["text", "label", "source"])
        df["label"] = df["label"].astype(int)
        df["source"] = df["source"].astype(str)
        df["text"] = df["text"].apply(lambda x: " ".join(str(x).split()[:256]))
        self.df = df
        print(f"Loaded {len(df)} valid records")
        print(f"Data distribution:")
        print(f"   Gold data: {len(df[df['source'] == 'gold'])} samples")
        print(f"   Weak data: {len(df[df['source'] == 'weak'])} samples")
        print(f"   Label distribution: {df['label'].value_counts().to_dict()}")

    def analyze_single_weight(self, weak_weight):
        """Trains and evaluates all models for a single given weak label weight."""
        df = self.df.copy()
        sample_weights = df["source"].apply(lambda x: weak_weight if x == "weak" else 1.0)
        X = df["text"]
        y = df["label"]
        source = df["source"]

        X_vec = self.vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X_vec, y, source, test_size=0.2, stratify=y, random_state=42
        )

        weight_train = sample_weights.iloc[y_train.index]

        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train, sample_weight=weight_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Calculate source-specific performance
            source_performance = {}
            for src in s_test.unique():
                src_mask = s_test == src
                if src_mask.sum() > 0:
                    source_performance[src] = {
                        "accuracy": accuracy_score(y_test[src_mask], y_pred[src_mask]),
                        "count": src_mask.sum()
                    }

            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "auc": roc_auc_score(y_test, y_proba) if y_proba is not None else 0,
                "cv_mean": cross_val_score(model, X_vec, y, cv=5, scoring="accuracy").mean(),
                "source_performance": source_performance
            }
        self.results_by_weight[weak_weight] = results

    def run_multi_weight_analysis(self, weight_list):
        """Iterates through a list of weights, running the analysis for each."""
        print("\n" + "="*50)
        print("Step 2: Running multi-weight analysis")
        print("="*50)
        
        for weight in weight_list:
            print(f"\nAnalyzing weak label weight = {weight}")
            self.analyze_single_weight(weight)
            
            print(f"Results for weight {weight}:")
            print(f"{'Model':<20} {'Accuracy':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'AUC':<8} {'CV':<8}")
            print("-" * 80)
            
            for model_name, metrics in self.results_by_weight[weight].items():
                print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<8.4f} "
                      f"{metrics['precision']:<10.4f} {metrics['recall']:<8.4f} "
                      f"{metrics['auc']:<8.4f} {metrics['cv_mean']:<8.4f}")
            
            best_model = max(self.results_by_weight[weight].keys(), 
                           key=lambda k: self.results_by_weight[weight][k]['cv_mean'])
            best_score = self.results_by_weight[weight][best_model]['cv_mean']
            print(f"Best model: {best_model} (CV Score: {best_score:.4f})")
            
        print("\nAll weights analyzed successfully")

    def create_comparison_plots(self):
        """Generates and saves plots comparing model performance across weights."""
        print("\n" + "="*50)
        print("Step 3: Generating performance comparison plots")
        print("="*50)
        
        weights = list(self.results_by_weight.keys())
        model_names = list(list(self.results_by_weight.values())[0].keys())

        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle('Model Performance Across Different Weak Label Weights', fontsize=20, fontweight='bold')

        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc', 'cv_mean']
        metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC', 'Cross-Validation Accuracy']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            row, col = divmod(idx, 2)
            ax = axes[row][col]
            for i, model_name in enumerate(model_names):
                values = [self.results_by_weight[w][model_name][metric] for w in weights]
                ax.plot(weights, values, marker='o', linewidth=2, label=model_name, color=colors[i])
            ax.set_xlabel('Weak Label Weight', fontsize=14)
            ax.set_ylabel(metric_name, fontsize=14)
            ax.set_title(f'{metric_name} vs Weak Label Weight', fontsize=16)
            ax.set_ylim(0.5, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('weight_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Plot saved: weight_comparison_analysis.png")

    def create_source_performance_plot(self):
        """Generates plots comparing performance on gold vs. weak data subsets."""
        print("\n" + "="*50)
        print("Step 4: Generating source performance comparison")
        print("="*50)
    
        weights = list(self.results_by_weight.keys())
        model_names = list(list(self.results_by_weight.values())[0].keys())

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Accuracy on Gold vs Weak Data Under Different Weights', fontsize=18, fontweight='bold')

        for i, model_name in enumerate(model_names):
            ax = axes[i]
            gold_accs = [self.results_by_weight[w][model_name]['source_performance'].get('gold', {}).get('accuracy', 0) for w in weights]
            weak_accs = [self.results_by_weight[w][model_name]['source_performance'].get('weak', {}).get('accuracy', 0) for w in weights]

            ax.plot(weights, gold_accs, marker='o', label='Gold Data', color='gold', linewidth=2)
            ax.plot(weights, weak_accs, marker='s', label='Weak Label Data', color='skyblue', linewidth=2)
            ax.set_title(model_name, fontsize=14)
            ax.set_xlabel('Weak Label Weight', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_ylim(0.5, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('source_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Plot saved: source_performance_comparison.png")

    def export_summary_csv(self):
        """Exports all collected metrics to a CSV file and prints the best result."""
        print("\n" + "="*50)
        print("Step 5: Exporting result summary")
        print("="*50)
        summary_rows = []
        for weight, model_results in self.results_by_weight.items():
            for model_name, metrics in model_results.items():
                row = {"Weight": weight, "Model": model_name, **metrics}
                summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        # Select and reorder columns for clarity
        cols = ["Weight", "Model", "Accuracy", "F1 Score", "Precision", "Recall", "AUC", "Cross-Validation Accuracy"]
        df_summary = df_summary.reindex(columns=cols).rename(columns={"cv_mean": "Cross-Validation Accuracy", "f1_score": "F1 Score", "accuracy": "Accuracy", "precision": "Precision", "recall":"Recall", "auc":"AUC"})

        df_summary.to_csv("weight_analysis_report.csv", index=False)
        print("Summary exported: weight_analysis_report.csv")
        
        best_row = df_summary.loc[df_summary['Cross-Validation Accuracy'].idxmax()]
        print(f"\nOverall Best Configuration:")
        print(f"   Weight: {best_row['Weight']}")
        print(f"   Model: {best_row['Model']}")
        print(f"   Accuracy: {best_row['Accuracy']:.4f}")
        print(f"   F1 Score: {best_row['F1 Score']:.4f}")
        print(f"   CV Accuracy: {best_row['Cross-Validation Accuracy']:.4f}")

    def run_complete_analysis(self, weight_list=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
        """Runs the complete analysis pipeline from data loading to result export."""
        print("Starting Multi-Weight Text Classification Analysis")
        print("=" * 60)
        
        try:
            self.load_and_prepare_data()
            self.run_multi_weight_analysis(weight_list)
            self.create_comparison_plots()
            self.create_source_performance_plot()
            self.export_summary_csv()
            
            print("\n" + "="*60)
            print("Analysis Complete!")
            print("="*60)
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()

def quick_analysis(file_path, weight_list=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
    """A helper function to quickly run the full analysis on a given file."""
    classifier = TraditionalTextClassifier(file_path)
    classifier.run_complete_analysis(weight_list)
    return classifier

def prepare_data_from_separate_files(gold_file, weak_file, output_file):
    """Combines separate gold and weak label files into a single file."""
    print("Combining separate files...")
    
    gold_df = pd.read_excel(gold_file) if gold_file.endswith('.xlsx') else pd.read_csv(gold_file)
    weak_df = pd.read_excel(weak_file) if weak_file.endswith('.xlsx') else pd.read_csv(weak_file)
    
    gold_df['source'] = 'gold'
    weak_df['source'] = 'weak'
    
    combined_df = pd.concat([gold_df, weak_df], ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    
    print(f"Combined data saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Example usage:
    # It will automatically try to run if it finds 'labelsleuth.xlsx' and 'converted_data.csv'.
    import os
    if os.path.exists('labelsleuth.xlsx') and os.path.exists('converted_data.csv'):
        print("\nFound example files, running analysis...")
        
        combined_file = prepare_data_from_separate_files(
            'labelsleuth.xlsx', 
            'converted_data.csv', 
            'combined_analysis_data.csv'
        )
        
        classifier = quick_analysis(combined_file)
    else:
        print("\nTo run an analysis, prepare your data and call the functions.")
        print("Example: quick_analysis('your_combined_data.csv')")
