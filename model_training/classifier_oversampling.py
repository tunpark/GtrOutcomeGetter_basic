import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class SamplingStrategyTester:
    """
    Compares oversampling strategies for training on imbalanced text data.

    This class uses a pre-determined best weight for weak labels as a baseline
    and tests strategies like SMOTE and RandomOversampler to see if they can
    improve model performance, especially for the minority class.
    """
    
    def __init__(self, file_path, best_weight=0.05):
        self.file_path = file_path
        self.best_weight = best_weight
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "Linear SVM": LinearSVC()
        }
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.results = {}
        
    def load_and_prepare_data(self):
        """Loads data and applies the best weight configuration from previous analysis."""
        print("="*50)
        print("Loading data and applying best weight configuration")
        print("="*50)
        
        df = pd.read_csv(self.file_path)
        df = df.dropna(subset=["text", "label", "source"])
        df["label"] = df["label"].astype(int)
        df["text"] = df["text"].apply(lambda x: " ".join(str(x).split()[:256]))
        
        # Apply the pre-determined best weight for weak labels
        sample_weights = df["source"].apply(lambda x: self.best_weight if x == "weak" else 1.0)
        df['sample_weight'] = sample_weights
        
        self.df = df
        
        print(f"Loaded {len(df)} valid records")
        print(f"Using best weight for weak labels: {self.best_weight}")
        
        class_dist = df['label'].value_counts().sort_index()
        print(f"Class distribution: {dict(class_dist)}")
        if class_dist.min() > 0:
            imbalance_ratio = class_dist.max() / class_dist.min()
            print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
        
    def test_sampling_strategies(self):
        """Tests and compares several sampling strategies for imbalanced data."""
        print("\n" + "="*50)
        print("Testing Sampling Strategies for Highly Imbalanced Data")
        print("="*50)
        
        X = self.df["text"]
        y = self.df["label"]
        weights = self.df["sample_weight"]
        X_vec = self.vectorizer.fit_transform(X)
        
        X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
            X_vec, y, weights, test_size=0.2, stratify=y, random_state=42
        )
        
        print(f"Original training set distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        # Define sampling strategies
        # A check is added for k_neighbors in SMOTE for highly imbalanced cases
        k_neighbors_val = min(5, np.sum(y_train == 1) - 1 if np.sum(y_train == 1) > 1 else 1)

        sampling_strategies = {
            "Weighted Only": None,
            "Random Oversampling": RandomOverSampler(random_state=42),
            "SMOTE": SMOTE(random_state=42, k_neighbors=k_neighbors_val),
            "SMOTE + Tomek": SMOTETomek(random_state=42, 
                                      smote=SMOTE(random_state=42, k_neighbors=k_neighbors_val))
        }
        
        for strategy_name, sampler in sampling_strategies.items():
            print(f"\n--- Testing {strategy_name} ---")
            
            X_train_resampled, y_train_resampled = X_train, y_train
            weights_resampled = w_train if sampler is None else None

            if sampler:
                try:
                    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
                    print(f"After {strategy_name}: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")
                except Exception as e:
                    print(f"Failed to apply {strategy_name}: {e}")
                    continue
            
            # Test all models with this sampling strategy
            strategy_results = {}
            for model_name, model in self.models.items():
                try:
                    # Train model
                    model.fit(X_train_resampled, y_train_resampled, sample_weight=weights_resampled)
                    y_pred = model.predict(X_test)
                    
                    # Store metrics
                    strategy_results[model_name] = self._calculate_metrics(y_test, y_pred, model, X_test)
                    
                    print(f"  {model_name}:")
                    print(f"    Overall F1: {strategy_results[model_name]['f1_score']:.4f}, AUC: {strategy_results[model_name]['auc']:.4f}")
                    print(f"    Minority Class F1: {strategy_results[model_name]['minority_f1']:.4f}, Recall: {strategy_results[model_name]['minority_recall']:.4f}")

                except Exception as e:
                    print(f"  {model_name}: Failed - {e}")
            
            self.results[strategy_name] = strategy_results

    def _calculate_metrics(self, y_true, y_pred, model, X_test):
        """Helper function to compute a dictionary of performance metrics."""
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        auc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0.0

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'auc': auc,
            'minority_precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            'minority_recall': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            'minority_f1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            'per_class_f1': per_class_f1
        }

    def create_comparison_plots(self):
        """Creates and saves plots comparing the sampling strategies."""
        print("\n" + "="*50)
        print("Creating comparison plots")
        print("="*50)
        
        strategies = list(self.results.keys())
        models = list(self.models.keys())
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sampling Strategy Comparison for Imbalanced Data', fontsize=16, fontweight='bold')
        
        metrics_to_plot = {
            'Overall F1 Score': 'f1_score',
            'Minority Class (Class 1) F1 Score': 'minority_f1',
            'Minority Class Recall': 'minority_recall',
            'AUC Score': 'auc'
        }

        for ax, (title, metric_key) in zip(axes.flatten(), metrics_to_plot.items()):
            self._plot_metric(ax, title, metric_key, strategies, models)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('imbalanced_sampling_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Plot saved: imbalanced_sampling_comparison.png")

    def _plot_metric(self, ax, title, metric_key, strategies, models):
        """Helper function to plot a single metric comparison chart."""
        x_pos = np.arange(len(strategies))
        width = 0.25
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, model in enumerate(models):
            values = [self.results[s].get(model, {}).get(metric_key, 0) for s in strategies]
            ax.bar(x_pos + i * width, values, width, label=model, color=colors[i], alpha=0.8)
        
        ax.set_title(title)
        ax.set_ylabel(title.split()[-1])
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def analyze_best_strategies(self):
        """Analyzes and prints a summary of the best performing strategies."""
        print("\n" + "="*50)
        print("Best Strategy Analysis")
        print("="*50)
        
        best_minority_f1 = 0
        best_overall_config = None
        
        for strategy, model_results in self.results.items():
            for model, metrics in model_results.items():
                if metrics['minority_f1'] > best_minority_f1:
                    best_minority_f1 = metrics['minority_f1']
                    best_overall_config = (strategy, model)
        
        if best_overall_config:
            strategy, model = best_overall_config
            result = self.results[strategy][model]
            
            print("OVERALL RECOMMENDATION (based on Minority F1 Score):")
            print(f"  Best Configuration: {model} with {strategy}")
            print(f"  Minority Class F1: {result['minority_f1']:.4f} (Recall: {result['minority_recall']:.4f}, Precision: {result['minority_precision']:.4f})")
            print(f"  Overall F1: {result['f1_score']:.4f}, AUC: {result['auc']:.4f}")
            
            if 'Weighted Only' in self.results and model in self.results['Weighted Only']:
                baseline_f1 = self.results['Weighted Only'][model]['minority_f1']
                improvement = result['minority_f1'] - baseline_f1
                print(f"  Improvement over weighted-only baseline: +{improvement:.4f} in minority F1")

    def export_results(self):
        """Exports detailed performance metrics to a CSV file."""
        print("\n" + "="*50)
        print("Exporting results")
        print("="*50)
        
        rows = []
        for strategy, models in self.results.items():
            for model, metrics in models.items():
                row = {'Sampling_Strategy': strategy, 'Model': model, **metrics}
                rows.append(row)
        
        df_results = pd.DataFrame(rows)
        df_results.to_csv('imbalanced_sampling_results.csv', index=False)
        print("Results exported: imbalanced_sampling_results.csv")
        return df_results
    
    def run_complete_test(self):
        """Runs the complete analysis pipeline for sampling strategies."""
        print("Starting Imbalanced Data Sampling Analysis")
        print("=" * 60)
        
        try:
            self.load_and_prepare_data()
            self.test_sampling_strategies()
            self.create_comparison_plots()
            self.analyze_best_strategies()
            self.export_results()
            
            print("\n" + "="*60)
            print("Imbalanced Data Sampling Analysis Complete!")
            print("="*60)
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()

def quick_imbalanced_test(file_path, best_weight=0.05):
    """A helper function to quickly run the full analysis."""
    tester = SamplingStrategyTester(file_path, best_weight)
    tester.run_complete_test()

if __name__ == "__main__":
    import os
    if os.path.exists('combined_analysis_data.csv'):
        print("\nFound combined data file! Running imbalanced data analysis...")
        quick_imbalanced_test('combined_analysis_data.csv', best_weight=0.05)
    else:
        print("\nTo run the test, ensure 'combined_analysis_data.csv' exists.")
        print("Example: quick_imbalanced_test('combined_analysis_data.csv', best_weight=0.05)")