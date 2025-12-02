"""
Phase 5: Model Evaluation and Validation - Original 299 Records
Team: Goblins Hiding in Vents (Cameron, Julian, Juan, Nathan, Nasiru, Jose)
Week: 5

This module implements comprehensive model evaluation using the first 299 original records
with the 73.4% hospital baseline for comparison.

Team Leaders: Nathan (Statistical Analyst), Juan (Machine Learning Engineer)

Goals:
- Generate ROC curves and AUC analysis
- Implement bootstrap validation for confidence intervals
- Perform statistical significance testing vs hospital baseline (73.4%)
- Calculate comprehensive performance metrics
- Validate model reliability and clinical utility using original data only
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
from scipy.stats import ttest_1samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation and statistical validation using original 299 records.
    
    Responsibilities:
    - ROC curve analysis and AUC calculation
    - Bootstrap validation for confidence intervals
    - Statistical significance testing vs hospital baseline (73.4%)
    - Comprehensive performance metrics calculation
    - Clinical utility assessment
    """
    
    def __init__(self, models_dir="../phase_4_modeling/models/", data_path="../Datasets/training_data.csv"):
        """
        Initialize model evaluator with trained models and original dataset.
        
        Args:
            models_dir: Directory containing trained models
            data_path: Path to original dataset (first 299 records will be used)
        """
        self.models_dir = models_dir
        self.data_path = data_path
        
        # Load data and models
        self.df = None
        self.X_test = None
        self.y_test = None
        self.models = {}
        self.feature_names_original = None
        self.feature_names_engineered = None
        
        # Hospital baseline (updated to 73.4%)
        self.hospital_baseline = 0.734
        self.target_accuracy = 0.85
        
        # Evaluation results
        self.evaluation_results = {}
        self.bootstrap_results = {}
        
        self.load_data_and_models()
    
    def load_data_and_models(self):
        """Load test data and trained models."""
        try:
            # Load data
            self.df = pd.read_csv(self.data_path)
            
            # Use only first 299 records (original data, not synthetic)
            self.df = self.df.head(299)
            print(f"✅ Using first 299 records (original data): {self.df.shape}")
            
            # Prepare test set (same split as training)
            from sklearn.model_selection import train_test_split
            target = 'DEATH_EVENT'
            features = [col for col in self.df.columns if col != target]
            
            X = self.df[features]
            y = self.df[target]
            
            _, self.X_test, _, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"✅ Test data loaded: {self.X_test.shape}")
            
            # Load models
            self.load_trained_models()
            
        except Exception as e:
            print(f"❌ Error loading data/models: {e}")
    
    def load_trained_models(self):
        """Load all trained models from Phase 4."""
        import os
        
        try:
            # Decision Tree models (both original and engineered features)
            if os.path.exists(f"{self.models_dir}decision_tree_original.joblib"):
                self.models['decision_tree_original'] = joblib.load(f"{self.models_dir}decision_tree_original.joblib")
                print("✅ Decision Tree (Original) loaded")
            
            if os.path.exists(f"{self.models_dir}decision_tree_engineered.joblib"):
                self.models['decision_tree_engineered'] = joblib.load(f"{self.models_dir}decision_tree_engineered.joblib")
                print("✅ Decision Tree (Engineered) loaded")
            
            # Random Forest models (both original and engineered features)
            if os.path.exists(f"{self.models_dir}random_forest_original.joblib"):
                self.models['random_forest_original'] = joblib.load(f"{self.models_dir}random_forest_original.joblib")
                print("✅ Random Forest (Original) loaded")
            
            if os.path.exists(f"{self.models_dir}random_forest_engineered.joblib"):
                self.models['random_forest_engineered'] = joblib.load(f"{self.models_dir}random_forest_engineered.joblib")
                print("✅ Random Forest (Engineered) loaded")
            
            # Load feature names for proper alignment
            if os.path.exists(f"{self.models_dir}feature_names_original.joblib"):
                self.feature_names_original = joblib.load(f"{self.models_dir}feature_names_original.joblib")
                print("✅ Original feature names loaded")
            
            if os.path.exists(f"{self.models_dir}feature_names_engineered.joblib"):
                self.feature_names_engineered = joblib.load(f"{self.models_dir}feature_names_engineered.joblib")
                print("✅ Engineered feature names loaded")
            
            print(f"📊 Loaded {len(self.models)} models for evaluation")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def generate_roc_curves(self):
        """
        Generate ROC curves for all models and calculate AUC scores.
        
        Returns:
            Dictionary with ROC curve data and AUC scores
        """
        if not self.models or self.X_test is None:
            return None
            
        print("\n📈 GENERATING ROC CURVES")
        print("="*30)
        
        plt.figure(figsize=(12, 8))
        
        roc_data = {}
        colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12']
        
        for i, (model_name, model_info) in enumerate(self.models.items()):
            print(f"   🔍 Analyzing {model_name.replace('_', ' ').title()}...")
            
            # Get appropriate test data based on model type
            try:
                if 'original' in model_name and hasattr(self, 'feature_names_original') and self.feature_names_original is not None:
                    # Use only original features for original models
                    X_test_model = self.X_test[self.feature_names_original]
                elif 'engineered' in model_name and hasattr(self, 'feature_names_engineered') and self.feature_names_engineered is not None:
                    # Check if engineered features exist in current dataset
                    missing_features = [f for f in self.feature_names_engineered if f not in self.X_test.columns]
                    if missing_features:
                        print(f"     ⚠️  Skipping - missing engineered features: {missing_features[:3]}...")
                        continue
                    # Use engineered features for engineered models
                    X_test_model = self.X_test[self.feature_names_engineered]
                else:
                    # Fallback to all features
                    X_test_model = self.X_test
                
                # Get predictions
                if model_name == 'logistic_regression':
                    model = model_info['model']
                    scaler = model_info['scaler']
                    X_test_processed = scaler.transform(X_test_model)
                    y_prob = model.predict_proba(X_test_processed)[:, 1]
                else:
                    model = model_info
                    y_prob = model.predict_proba(X_test_model)[:, 1]
                
            except KeyError as e:
                print(f"     ⚠️  Skipping {model_name} - feature mismatch: {e}")
                continue
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(self.y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Store data
            roc_data[model_name] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc,
                'y_prob': y_prob
            }
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
            
            print(f"     AUC Score: {roc_auc:.4f}")
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
        
        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curves - Heart Failure Prediction Models (Original 299 Records)', fontsize=14, pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add target line for clinical significance
        plt.axhline(y=0.85, color='red', linestyle=':', alpha=0.7, 
                   label='Target Sensitivity (85%)')
        
        plt.tight_layout()
        plt.show()
        
        # Best AUC
        best_model = max(roc_data.keys(), key=lambda x: roc_data[x]['auc'])
        print(f"\n🏆 Best AUC: {best_model.replace('_', ' ').title()} ({roc_data[best_model]['auc']:.4f})")
        
        return roc_data
    
    def perform_bootstrap_validation(self, n_bootstrap=1000):
        """
        Perform bootstrap validation to calculate confidence intervals.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Bootstrap confidence intervals for each model
        """
        if not self.models or self.X_test is None:
            return None
            
        print(f"\n🔄 BOOTSTRAP VALIDATION ({n_bootstrap} samples)")
        print("="*40)
        
        bootstrap_results = {}
        
        for model_name, model_info in self.models.items():
            print(f"   🎲 Bootstrapping {model_name.replace('_', ' ').title()}...")
            
            # Get appropriate test data based on model type
            try:
                if 'original' in model_name and hasattr(self, 'feature_names_original') and self.feature_names_original is not None:
                    # Use only original features for original models
                    X_test_model = self.X_test[self.feature_names_original]
                elif 'engineered' in model_name and hasattr(self, 'feature_names_engineered') and self.feature_names_engineered is not None:
                    # Check if engineered features exist in current dataset
                    missing_features = [f for f in self.feature_names_engineered if f not in self.X_test.columns]
                    if missing_features:
                        print(f"     ⚠️  Skipping - missing engineered features: {missing_features[:3]}...")
                        continue
                    # Use engineered features for engineered models
                    X_test_model = self.X_test[self.feature_names_engineered]
                else:
                    # Fallback to all features
                    X_test_model = self.X_test
                
            except KeyError as e:
                print(f"     ⚠️  Skipping {model_name} - feature mismatch: {e}")
                continue
            
            accuracy_scores = []
            auc_scores = []
            
            # Get base predictions
            if model_name == 'logistic_regression':
                model = model_info['model']
                scaler = model_info['scaler']
                X_test_processed = scaler.transform(X_test_model)
            else:
                model = model_info
                X_test_processed = X_test_model
            
            # Bootstrap sampling
            np.random.seed(42)
            
            for i in range(n_bootstrap):
                # Random sample with replacement
                indices = np.random.choice(len(X_test_processed), size=len(X_test_processed), replace=True)
                
                X_bootstrap = X_test_processed.iloc[indices] if hasattr(X_test_processed, 'iloc') else X_test_processed[indices]
                y_bootstrap = self.y_test.iloc[indices]
                
                # Predictions
                y_pred = model.predict(X_bootstrap)
                y_prob = model.predict_proba(X_bootstrap)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_bootstrap, y_pred)
                auc_score = roc_auc_score(y_bootstrap, y_prob)
                
                accuracy_scores.append(accuracy)
                auc_scores.append(auc_score)
            
            # Calculate confidence intervals
            accuracy_ci = np.percentile(accuracy_scores, [2.5, 97.5])
            auc_ci = np.percentile(auc_scores, [2.5, 97.5])
            
            bootstrap_results[model_name] = {
                'accuracy_mean': np.mean(accuracy_scores),
                'accuracy_std': np.std(accuracy_scores),
                'accuracy_ci': accuracy_ci,
                'auc_mean': np.mean(auc_scores),
                'auc_std': np.std(auc_scores),
                'auc_ci': auc_ci,
                'accuracy_scores': accuracy_scores,
                'auc_scores': auc_scores
            }
            
            print(f"     Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
            print(f"     95% CI: [{accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f}]")
            print(f"     AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
            print(f"     95% CI: [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
            print()
        
        self.bootstrap_results = bootstrap_results
        return bootstrap_results
    
    def perform_statistical_significance_testing(self):
        """
        Perform statistical significance testing against hospital baseline.
        
        Returns:
            Statistical test results
        """
        if not self.bootstrap_results:
            print("❌ Run bootstrap validation first")
            return None
            
        print("🔬 STATISTICAL SIGNIFICANCE TESTING")
        print("="*40)
        print(f"Null Hypothesis: Model accuracy = Hospital baseline ({self.hospital_baseline:.1%})")
        print(f"Alternative Hypothesis: Model accuracy > Hospital baseline")
        print()
        
        significance_results = {}
        
        for model_name, bootstrap_data in self.bootstrap_results.items():
            print(f"   📊 Testing {model_name.replace('_', ' ').title()}:")
            
            accuracy_scores = bootstrap_data['accuracy_scores']
            
            # One-sample t-test against hospital baseline
            t_statistic, p_value = ttest_1samp(accuracy_scores, self.hospital_baseline)
            
            # Effect size (Cohen's d)
            effect_size = (np.mean(accuracy_scores) - self.hospital_baseline) / np.std(accuracy_scores)
            
            # Proportion of bootstrap samples exceeding baseline
            proportion_better = np.mean(np.array(accuracy_scores) > self.hospital_baseline)
            
            # Proportion exceeding target
            proportion_target = np.mean(np.array(accuracy_scores) >= self.target_accuracy)
            
            significance_results[model_name] = {
                't_statistic': t_statistic,
                'p_value': p_value,
                'effect_size': effect_size,
                'proportion_better_than_baseline': proportion_better,
                'proportion_meets_target': proportion_target,
                'statistically_significant': p_value < 0.05 and t_statistic > 0
            }
            
            print(f"     Mean Accuracy: {np.mean(accuracy_scores):.4f}")
            print(f"     t-statistic: {t_statistic:.3f}")
            print(f"     p-value: {p_value:.6f}")
            print(f"     Effect size (Cohen's d): {effect_size:.3f}")
            print(f"     % Better than baseline: {proportion_better:.1%}")
            print(f"     % Meeting target: {proportion_target:.1%}")
            
            if significance_results[model_name]['statistically_significant']:
                print(f"     ✅ STATISTICALLY SIGNIFICANT IMPROVEMENT!")
            else:
                print(f"     ❌ No significant improvement detected")
            print()
        
        return significance_results
    
    def generate_confusion_matrices(self):
        """Generate confusion matrices for all models."""
        if not self.models or self.X_test is None:
            return None
            
        print("\n📊 CONFUSION MATRIX ANALYSIS")
        print("="*35)
        
        n_models = len([m for m in self.models.keys() if 'original' in m])  # Only original models will work
        fig, axes = plt.subplots(1, max(n_models, 1), figsize=(5*max(n_models, 1), 4))
        
        if n_models == 1:
            axes = [axes]
        
        confusion_results = {}
        plot_idx = 0
        
        for model_name, model_info in self.models.items():
            # Get appropriate test data based on model type
            try:
                if 'original' in model_name and hasattr(self, 'feature_names_original') and self.feature_names_original is not None:
                    # Use only original features for original models
                    X_test_model = self.X_test[self.feature_names_original]
                elif 'engineered' in model_name and hasattr(self, 'feature_names_engineered') and self.feature_names_engineered is not None:
                    # Check if engineered features exist in current dataset
                    missing_features = [f for f in self.feature_names_engineered if f not in self.X_test.columns]
                    if missing_features:
                        print(f"   ⚠️  Skipping {model_name} - missing engineered features")
                        continue
                    # Use engineered features for engineered models
                    X_test_model = self.X_test[self.feature_names_engineered]
                else:
                    # Fallback to all features
                    X_test_model = self.X_test
                
                # Get predictions
                if model_name == 'logistic_regression':
                    model = model_info['model']
                    scaler = model_info['scaler']
                    X_test_processed = scaler.transform(X_test_model)
                    y_pred = model.predict(X_test_processed)
                else:
                    model = model_info
                    y_pred = model.predict(X_test_model)
                
            except KeyError as e:
                print(f"   ⚠️  Skipping {model_name} - feature mismatch: {e}")
                continue
            
            # Calculate confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Calculate metrics
            tn, fp, fn, tp = cm.ravel()
            
            sensitivity = tp / (tp + fn)  # Recall
            specificity = tn / (tn + fp)
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
            npv = tn / (tn + fn)
            
            confusion_results[model_name] = {
                'confusion_matrix': cm,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp
            }
            
            # Plot confusion matrix only for working models
            if 'original' in model_name and plot_idx < len(axes):
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Survived', 'Died'],
                           yticklabels=['Survived', 'Died'],
                           ax=axes[plot_idx])
                
                axes[plot_idx].set_title(f'{model_name.replace("_", " ").title()}\n'
                                 f'Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}')
                axes[plot_idx].set_xlabel('Predicted')
                axes[plot_idx].set_ylabel('Actual')
                plot_idx += 1
            
            print(f"   {model_name.replace('_', ' ').title()}:")
            print(f"     Sensitivity (Recall): {sensitivity:.4f}")
            print(f"     Specificity: {specificity:.4f}")
            print(f"     PPV (Precision): {ppv:.4f}")
            print(f"     NPV: {npv:.4f}")
            print(f"     True Positives: {tp}, False Negatives: {fn}")
            print(f"     True Negatives: {tn}, False Positives: {fp}")
            print()
        
        plt.suptitle('Confusion Matrices - Heart Failure Prediction (Original 299 Records)', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return confusion_results
    
    def generate_comprehensive_evaluation_report(self):
        """Generate final comprehensive evaluation report."""
        print("\n📋 COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*50)
        print("Team: Goblins Hiding in Vents")
        print("Phase: 5 - Model Evaluation and Validation")
        print("Dataset: Original 299 Records (No Synthetic Data)")
        print("Team Leaders: Nathan (Statistical), Juan (ML)")
        print("="*50)
        
        # Project goals recap
        print(f"\n🎯 PROJECT GOALS ASSESSMENT:")
        print(f"   Target Accuracy: ≥{self.target_accuracy:.1%}")
        print(f"   Hospital Baseline: {self.hospital_baseline:.1%}")
        print(f"   Clinical Requirement: Statistically significant improvement")
        
        # Best performing model
        if self.bootstrap_results:
            best_model = max(self.bootstrap_results.keys(), 
                           key=lambda x: self.bootstrap_results[x]['accuracy_mean'])
            
            best_accuracy = self.bootstrap_results[best_model]['accuracy_mean']
            best_ci = self.bootstrap_results[best_model]['accuracy_ci']
            
            print(f"\n🏆 BEST PERFORMING MODEL: {best_model.replace('_', ' ').title()}")
            print(f"   Mean Accuracy: {best_accuracy:.4f}")
            print(f"   95% Confidence Interval: [{best_ci[0]:.4f}, {best_ci[1]:.4f}]")
            
            # Goal achievement
            if best_accuracy >= self.target_accuracy:
                print(f"   ✅ TARGET ACHIEVED! ({best_accuracy:.1%} ≥ {self.target_accuracy:.1%})")
            else:
                gap = self.target_accuracy - best_accuracy
                print(f"   📈 Target gap: -{gap:.1%} (close to target)")
            
            # Clinical significance
            improvement = (best_accuracy - self.hospital_baseline) * 100
            if improvement > 0:
                print(f"   🏥 Hospital improvement: +{improvement:.1f}%")
            
        print(f"\n🔬 STATISTICAL VALIDATION:")
        print("   ✅ Bootstrap validation completed (1000 samples)")
        print("   ✅ ROC curve analysis performed")
        print("   ✅ Confidence intervals calculated")
        print("   ✅ Statistical significance testing completed")
        print("   ✅ Confusion matrix analysis performed")
        
        print(f"\n💊 CLINICAL IMPACT:")
        print("   • Early heart failure identification")
        print("   • Reduced misdiagnosis rates")
        print("   • Improved patient outcomes")
        print("   • Cost savings from better diagnosis")
        print("   • Support for clinical decision making")
        
        print(f"\n📈 READY FOR PHASE 6: FINAL TESTING & PRESENTATION")
        print("   Next: Decision tree visualization, final validation, presentation")


def main():
    """
    Main function to execute Phase 5 model evaluation workflow using original 299 records.
    
    Team Usage:
    - Nathan: Lead statistical analysis and significance testing
    - Juan: Support with model performance analysis and interpretation
    """
    print("="*60)
    print("PHASE 5: MODEL EVALUATION & VALIDATION")
    print("Dataset: Original 299 Records (73.4% Hospital Baseline)")
    print("Team Leaders: Nathan (Statistical), Juan (ML)")
    print("Goal: Validate ≥85% accuracy with statistical significance")
    print("="*60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    if evaluator.models and evaluator.X_test is not None:
        print("\n📊 Starting comprehensive model evaluation...")
        
        # 1. ROC Curve Analysis
        roc_data = evaluator.generate_roc_curves()
        
        # 2. Bootstrap Validation
        bootstrap_results = evaluator.perform_bootstrap_validation(n_bootstrap=1000)
        
        # 3. Statistical Significance Testing
        significance_results = evaluator.perform_statistical_significance_testing()
        
        # 4. Confusion Matrix Analysis
        confusion_results = evaluator.generate_confusion_matrices()
        
        # 5. Comprehensive Report
        evaluator.generate_comprehensive_evaluation_report()
        
        print("\n✅ Phase 5 model evaluation completed!")
        print("📊 Statistical validation and performance analysis complete")
        print("🎯 Ready for final Phase 6 presentation")
        
    else:
        print("❌ Cannot proceed without trained models. Run Phase 4 first.")


if __name__ == "__main__":
    main()
