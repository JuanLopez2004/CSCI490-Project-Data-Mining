"""
Phase 4: Model Development and Training
Team: Goblins Hiding in Vents (Cameron, Julian, Juan, Nathan, Nasiru, Jose)
Week: 4

This module implements Decision Tree Classifier and other ML models for heart failure prediction.
Team Leaders: Juan (Machine Learning Engineer), Julian (Feature Engineering Specialist)

Goals:
- Implement Decision Tree Classifier with hyperparameter tuning
- Compare with other ML algorithms (Random Forest, Logistic Regression)
- Implement k-fold cross-validation
- Optimize model performance for ≥85% accuracy target
- Prepare models for comprehensive evaluation
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, cross_val_score, StratifiedKFold, 
    train_test_split, validation_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')

class HeartFailureModelTrainer:
    """
    Comprehensive machine learning model trainer for heart failure prediction.
    
    Responsibilities:
    - Decision Tree Classifier implementation and optimization
    - Multiple algorithm comparison (Random Forest, Logistic Regression)
    - Hyperparameter tuning using GridSearchCV
    - k-fold cross-validation for robust evaluation
    - Model persistence and management
    """
    
    def __init__(self, data_path: str = "../data/processed/heart_failure_combined_1000.csv"):
        """
        Initialize model trainer with processed dataset.
        
        Args:
            data_path: Path to processed dataset (1000 records)
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Models
        self.models = {}
        self.best_models = {}
        self.cv_scores = {}
        
        # Target accuracy goal
        self.target_accuracy = 0.85
        self.baseline_accuracy = 0.839  # Hospital baseline
        
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """Load processed data and prepare for model training."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Processed data loaded: {self.df.shape}")
            
            # Prepare features and target
            target = 'DEATH_EVENT'
            features = [col for col in self.df.columns if col != target]
            
            X = self.df[features]
            y = self.df[target]
            
            # Train-test split with stratification
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"   Training set: {self.X_train.shape}")
            print(f"   Test set: {self.X_test.shape}")
            print(f"   Feature count: {len(features)}")
            
        except FileNotFoundError:
            print(f"❌ Processed data not found. Run Phase 3 first.")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def train_decision_tree(self):
        """
        Train and optimize Decision Tree Classifier with hyperparameter tuning.
        
        Returns:
            Best DecisionTreeClassifier model
        """
        if self.X_train is None:
            return None
            
        print("\n🌳 TRAINING DECISION TREE CLASSIFIER")
        print("="*45)
        
        # Decision Tree hyperparameter grid
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2', None]
        }
        
        print(f"   🔍 Hyperparameter search space: {len(param_grid['criterion']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} combinations")
        
        # Initialize base model
        dt_model = DecisionTreeClassifier(random_state=42)
        
        # GridSearchCV with cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=dt_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Train and find best parameters
        print("   🚀 Starting hyperparameter optimization...")
        grid_search.fit(self.X_train, self.y_train)
        
        # Store best model
        self.best_models['decision_tree'] = grid_search.best_estimator_
        
        print(f"\n   ✅ Best Decision Tree parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"     {param}: {value}")
        
        print(f"\n   📊 Cross-validation score: {grid_search.best_score_:.4f}")
        
        # Test set evaluation
        test_accuracy = grid_search.best_estimator_.score(self.X_test, self.y_test)
        print(f"   🎯 Test set accuracy: {test_accuracy:.4f}")
        
        # Check if target achieved
        if test_accuracy >= self.target_accuracy:
            print(f"   🏆 TARGET ACHIEVED! {test_accuracy:.1%} ≥ {self.target_accuracy:.1%}")
        else:
            print(f"   📈 Progress: {test_accuracy:.1%} (target: {self.target_accuracy:.1%})")
        
        return grid_search.best_estimator_
    
    def train_random_forest(self):
        """
        Train Random Forest for comparison with Decision Tree.
        
        Returns:
            Best RandomForestClassifier model
        """
        if self.X_train is None:
            return None
            
        print("\n🌲 TRAINING RANDOM FOREST CLASSIFIER")
        print("="*42)
        
        # Random Forest hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['sqrt', 'log2']
        }
        
        print(f"   🔍 Hyperparameter combinations: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])}")
        
        # Initialize base model
        rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # GridSearchCV
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Train
        print("   🚀 Training Random Forest...")
        grid_search.fit(self.X_train, self.y_train)
        
        # Store best model
        self.best_models['random_forest'] = grid_search.best_estimator_
        
        print(f"\n   ✅ Best Random Forest parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"     {param}: {value}")
        
        print(f"\n   📊 Cross-validation score: {grid_search.best_score_:.4f}")
        
        test_accuracy = grid_search.best_estimator_.score(self.X_test, self.y_test)
        print(f"   🎯 Test set accuracy: {test_accuracy:.4f}")
        
        return grid_search.best_estimator_
    
    def train_logistic_regression(self):
        """
        Train Logistic Regression for baseline comparison.
        
        Returns:
            Best LogisticRegression model
        """
        if self.X_train is None:
            return None
            
        print("\n📈 TRAINING LOGISTIC REGRESSION")
        print("="*35)
        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Logistic Regression hyperparameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        # Initialize base model
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # GridSearchCV
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=lr_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Train
        print("   🚀 Training Logistic Regression...")
        grid_search.fit(X_train_scaled, self.y_train)
        
        # Store best model and scaler
        self.best_models['logistic_regression'] = {
            'model': grid_search.best_estimator_,
            'scaler': scaler
        }
        
        print(f"\n   ✅ Best Logistic Regression parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"     {param}: {value}")
        
        print(f"\n   📊 Cross-validation score: {grid_search.best_score_:.4f}")
        
        test_accuracy = grid_search.best_estimator_.score(X_test_scaled, self.y_test)
        print(f"   🎯 Test set accuracy: {test_accuracy:.4f}")
        
        return grid_search.best_estimator_
    
    def perform_model_comparison(self):
        """
        Compare all trained models and select the best performer.
        
        Returns:
            Dictionary with model comparison results
        """
        if not self.best_models:
            print("❌ No models trained yet. Train models first.")
            return None
            
        print("\n🏁 MODEL COMPARISON ANALYSIS")
        print("="*35)
        
        comparison_results = {}
        
        for model_name, model_info in self.best_models.items():
            print(f"\n   📊 Evaluating {model_name.replace('_', ' ').title()}:")
            
            # Handle different model types
            if model_name == 'logistic_regression':
                model = model_info['model']
                scaler = model_info['scaler']
                X_test_processed = scaler.transform(self.X_test)
            else:
                model = model_info
                X_test_processed = self.X_test
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            y_prob = model.predict_proba(X_test_processed)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc_roc = roc_auc_score(self.y_test, y_prob)
            
            # Cross-validation score
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            if model_name == 'logistic_regression':
                X_train_processed = scaler.transform(self.X_train)
                cv_scores = cross_val_score(model, X_train_processed, self.y_train, cv=cv, scoring='accuracy')
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
            
            # Store results
            comparison_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'target_achieved': accuracy >= self.target_accuracy,
                'beats_baseline': accuracy > self.baseline_accuracy
            }
            
            # Print results
            print(f"     Accuracy: {accuracy:.4f} ({'✅' if accuracy >= self.target_accuracy else '📈'})")
            print(f"     Precision: {precision:.4f}")
            print(f"     Recall: {recall:.4f}")
            print(f"     F1-Score: {f1:.4f}")
            print(f"     AUC-ROC: {auc_roc:.4f}")
            print(f"     CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            if accuracy >= self.target_accuracy:
                print(f"     🏆 TARGET ACHIEVED!")
            if accuracy > self.baseline_accuracy:
                improvement = (accuracy - self.baseline_accuracy) * 100
                print(f"     📈 Hospital improvement: +{improvement:.1f}%")
        
        # Find best model
        best_model_name = max(comparison_results.keys(), 
                             key=lambda x: comparison_results[x]['accuracy'])
        
        print(f"\n🏆 BEST MODEL: {best_model_name.replace('_', ' ').title()}")
        print(f"   Accuracy: {comparison_results[best_model_name]['accuracy']:.4f}")
        
        return comparison_results, best_model_name
    
    def save_models(self, output_dir="../models/"):
        """Save all trained models for future use."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 SAVING MODELS TO {output_dir}")
        print("="*25)
        
        for model_name, model_info in self.best_models.items():
            if model_name == 'logistic_regression':
                # Save both model and scaler
                model_path = f"{output_dir}{model_name}_model.joblib"
                scaler_path = f"{output_dir}{model_name}_scaler.joblib"
                
                joblib.dump(model_info['model'], model_path)
                joblib.dump(model_info['scaler'], scaler_path)
                
                print(f"   ✅ {model_name}: model + scaler saved")
            else:
                model_path = f"{output_dir}{model_name}.joblib"
                joblib.dump(model_info, model_path)
                
                print(f"   ✅ {model_name}: saved")
        
        # Save feature names
        feature_names = list(self.X_train.columns)
        joblib.dump(feature_names, f"{output_dir}feature_names.joblib")
        print(f"   ✅ Feature names saved")
    
    def generate_training_report(self):
        """Generate comprehensive training report."""
        print("\n📋 PHASE 4 MODEL TRAINING REPORT")
        print("="*40)
        print("Team: Goblins Hiding in Vents")
        print("Phase: 4 - Model Development")
        print("Team Leaders: Juan (ML Engineer), Julian (Feature Engineering)")
        print("="*40)
        
        print(f"\n🎯 PROJECT GOALS:")
        print(f"   Target Accuracy: ≥{self.target_accuracy:.1%}")
        print(f"   Hospital Baseline: {self.baseline_accuracy:.1%}")
        print(f"   Dataset Size: {len(self.df)} records (299 original + 701 synthetic)")
        print(f"   Features Used: {self.X_train.shape[1]}")
        
        print(f"\n🏆 ACHIEVEMENTS:")
        if self.best_models:
            best_accuracy = max([
                self.best_models[name].score(self.X_test, self.y_test) if name != 'logistic_regression'
                else self.best_models[name]['model'].score(self.best_models[name]['scaler'].transform(self.X_test), self.y_test)
                for name in self.best_models.keys()
            ])
            
            if best_accuracy >= self.target_accuracy:
                print(f"   ✅ Target achieved: {best_accuracy:.1%}")
            else:
                print(f"   📈 Best performance: {best_accuracy:.1%}")
            
            if best_accuracy > self.baseline_accuracy:
                improvement = (best_accuracy - self.baseline_accuracy) * 100
                print(f"   🏥 Hospital improvement: +{improvement:.1f}%")
        
        print(f"\n🔬 METHODOLOGICAL STRENGTHS:")
        print("   • Comprehensive hyperparameter optimization")
        print("   • 10-fold stratified cross-validation")
        print("   • Multiple algorithm comparison")
        print("   • Robust train-test splitting")
        print("   • Feature engineering from Phase 3")
        print("   • Synthetic data augmentation")
        
        print(f"\n📈 READY FOR PHASE 5: MODEL EVALUATION")
        print("   Next: ROC curves, bootstrap validation, statistical testing")


def main():
    """
    Main function to execute Phase 4 model training workflow.
    
    Team Usage:
    - Juan: Lead model development and hyperparameter tuning
    - Julian: Support with feature validation and model optimization
    """
    print("="*60)
    print("PHASE 4: MODEL DEVELOPMENT & TRAINING")
    print("Team Leaders: Juan (ML Engineer), Julian (Feature Engineering)")
    print("Goal: Achieve ≥85% accuracy with Decision Tree and comparisons")
    print("="*60)
    
    # Initialize model trainer
    trainer = HeartFailureModelTrainer()
    
    if trainer.X_train is not None:
        print("\n🤖 Starting comprehensive model training...")
        
        # Train all models
        dt_model = trainer.train_decision_tree()
        rf_model = trainer.train_random_forest()
        lr_model = trainer.train_logistic_regression()
        
        # Model comparison
        if trainer.best_models:
            comparison_results, best_model = trainer.perform_model_comparison()
            
            # Save models
            trainer.save_models()
            
            # Generate report
            trainer.generate_training_report()
            
            print("\n✅ Phase 4 model training completed!")
            print(f"🏆 Best model: {best_model.replace('_', ' ').title()}")
            print("📊 All models saved for Phase 5 evaluation")
        
    else:
        print("❌ Cannot proceed without processed data. Run Phases 1-3 first.")


if __name__ == "__main__":
    main()