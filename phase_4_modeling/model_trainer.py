"""
Phase 4: Model Development and Training
Team: Goblins Hiding in Vents (Cameron, Julian, Juan, Nathan, Nasiru, Jose)
Week: 4

This module implements Decision Tree Classifier and Random Forest models for heart failure prediction.
Trains models on both original (13 features) and engineered (20 features) datasets.
Team Leaders: Juan (Machine Learning Engineer), Julian (Feature Engineering Specialist)

Goals:
- Train Decision Tree and Random Forest on original dataset
- Train Decision Tree and Random Forest on engineered dataset
- Implement hyperparameter tuning with GridSearchCV
- Implement k-fold cross-validation
- Save all 4 trained models for Phase 5 evaluation
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, ParameterGrid
from tqdm.auto import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')


class GridSearchCVProgressBar(GridSearchCV):
    """
    GridSearchCV wrapper that displays a progress bar during training.

    Shows real-time progress as hyperparameter combinations are evaluated
    across cross-validation folds.
    """

    def fit(self, X, y=None, **fit_params):
        """Fit with progress bar showing each hyperparameter combination."""
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone
        import numpy as np

        # Get all parameter combinations
        param_combinations = list(ParameterGrid(self.param_grid))

        # Progress bar
        with tqdm(total=len(param_combinations), desc="GridSearchCV", unit="combo") as pbar:
            all_scores = []
            all_params = []

            for params in param_combinations:
                # Clone estimator and set parameters
                model = clone(self.estimator)
                model.set_params(**params)

                # Cross-validate this combination
                scores = cross_val_score(
                    model, X, y,
                    cv=self.cv,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs
                )

                all_scores.append(scores.mean())
                all_params.append(params)

                # Update progress bar
                pbar.update(1)

            # Find best parameters
            best_idx = np.argmax(all_scores)
            self.best_params_ = all_params[best_idx]
            self.best_score_ = all_scores[best_idx]

            # Store results for compatibility
            self.cv_results_ = {
                'mean_test_score': np.array(all_scores),
                'params': all_params
            }

            # Refit on full data with best parameters
            if self.refit:
                self.best_estimator_ = clone(self.estimator)
                self.best_estimator_.set_params(**self.best_params_)
                self.best_estimator_.fit(X, y, **fit_params)

            return self

class HeartFailureModelTrainer:
    """
    Machine learning model trainer for heart failure prediction.

    Trains models on both original and engineered feature sets to demonstrate
    the value of Phase 3's feature engineering work.

    Responsibilities:
    - Load original (13 features) and engineered (20 features) datasets
    - Train Decision Tree on both datasets
    - Train Random Forest on both datasets
    - Hyperparameter tuning using GridSearchCV
    - k-fold cross-validation for robust training
    - Save all models for Phase 5 evaluation
    """

    def __init__(self):
        """Initialize model trainer with both datasets from Phase 3."""
        # Dataset paths (from csci490/ directory)
        self.original_train_path = "phase_3_features/datasets/train_data.csv"
        self.original_test_path = "phase_3_features/datasets/test_data.csv"
        self.engineered_path = "phase_3_features/results/data_with_engineered_features.csv"

        # Datasets
        self.original_data = {}  # Will hold train/test for original features
        self.engineered_data = {}  # Will hold train/test for engineered features

        # Models storage
        self.trained_models = {}

        # Target and baseline
        self.target_accuracy = 0.85
        self.baseline_accuracy = 0.839

        # Load both datasets
        self.load_datasets()

    def load_datasets(self):
        """Load both original and engineered datasets from Phase 3."""
        print("=" * 60)
        print("LOADING DATASETS FROM PHASE 3")
        print("=" * 60)

        try:
            # Load original dataset (pre-split by Phase 3)
            train_df = pd.read_csv(self.original_train_path)
            test_df = pd.read_csv(self.original_test_path)

            target = 'DEATH_EVENT'

            # Original features
            features_original = [col for col in train_df.columns if col != target]

            self.original_data['X_train'] = train_df[features_original]
            self.original_data['y_train'] = train_df[target]
            self.original_data['X_test'] = test_df[features_original]
            self.original_data['y_test'] = test_df[target]
            self.original_data['features'] = features_original

            print(f"\n✅ Original dataset loaded (13 features)")
            print(f"   Train: {self.original_data['X_train'].shape}")
            print(f"   Test: {self.original_data['X_test'].shape}")
            print(f"   Features: {len(features_original)}")

        except Exception as e:
            print(f"❌ Error loading original dataset: {e}")
            self.original_data = None

        try:
            # Load engineered dataset
            df_engineered = pd.read_csv(self.engineered_path)

            features_engineered = [col for col in df_engineered.columns if col != target]

            X = df_engineered[features_engineered]
            y = df_engineered[target]

            # Split with same random_state as Phase 3
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self.engineered_data['X_train'] = X_train
            self.engineered_data['y_train'] = y_train
            self.engineered_data['X_test'] = X_test
            self.engineered_data['y_test'] = y_test
            self.engineered_data['features'] = features_engineered

            print(f"\n✅ Engineered dataset loaded (20 features)")
            print(f"   Train: {X_train.shape}")
            print(f"   Test: {X_test.shape}")
            print(f"   Features: {len(features_engineered)}")

            # Show engineered features
            original_features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                               'ejection_fraction', 'high_blood_pressure', 'platelets',
                               'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
            engineered_only = [f for f in features_engineered if f not in original_features]
            print(f"   Engineered features: {engineered_only}")

        except Exception as e:
            print(f"❌ Error loading engineered dataset: {e}")
            self.engineered_data = None

    def train_decision_tree(self, dataset_name, X_train, y_train):
        """
        Train Decision Tree with hyperparameter tuning.

        Args:
            dataset_name: 'original' or 'engineered'
            X_train: Training features
            y_train: Training labels

        Returns:
            Best Decision Tree model
        """
        print(f"\n🌳 TRAINING DECISION TREE ({dataset_name.upper()})")
        print("=" * 50)

        # Hyperparameter grid
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2', None]
        }

        n_combinations = (len(param_grid['criterion']) * len(param_grid['max_depth']) *
                         len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) *
                         len(param_grid['max_features']))

        print(f"   🔍 Testing {n_combinations} hyperparameter combinations")

        # Initialize model
        dt_model = DecisionTreeClassifier(random_state=42)

        # GridSearchCV with 10-fold CV and progress bar
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        grid_search = GridSearchCVProgressBar(
            estimator=dt_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,  # Parallelize CV folds within each combo
            verbose=0
        )

        # Train
        print("   🚀 Starting hyperparameter optimization...")
        grid_search.fit(X_train, y_train)

        # Results
        print(f"\n   ✅ Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"      {param}: {value}")

        print(f"\n   📊 Cross-validation accuracy: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def train_random_forest(self, dataset_name, X_train, y_train):
        """
        Train Random Forest with hyperparameter tuning.

        Args:
            dataset_name: 'original' or 'engineered'
            X_train: Training features
            y_train: Training labels

        Returns:
            Best Random Forest model
        """
        print(f"\n🌲 TRAINING RANDOM FOREST ({dataset_name.upper()})")
        print("=" * 50)

        # Hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['sqrt', 'log2']
        }

        n_combinations = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) *
                         len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) *
                         len(param_grid['max_features']))

        print(f"   🔍 Testing {n_combinations} hyperparameter combinations")

        # Initialize model
        rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

        # GridSearchCV with 10-fold CV and progress bar
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        grid_search = GridSearchCVProgressBar(
            estimator=rf_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,  # Parallelize CV folds within each combo
            verbose=0
        )

        # Train
        print("   🚀 Starting hyperparameter optimization...")
        grid_search.fit(X_train, y_train)

        # Results
        print(f"\n   ✅ Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"      {param}: {value}")

        print(f"\n   📊 Cross-validation accuracy: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def train_all_models(self):
        """Train all 4 models (2 algorithms × 2 datasets)."""
        if not self.original_data or not self.engineered_data:
            print("❌ Cannot train models without both datasets")
            return

        # 1. Decision Tree - Original
        dt_original = self.train_decision_tree(
            'original',
            self.original_data['X_train'],
            self.original_data['y_train']
        )
        self.trained_models['decision_tree_original'] = dt_original
        print("✅ Model 1/4 complete: Decision Tree (Original)\n")

        # 2. Decision Tree - Engineered
        dt_engineered = self.train_decision_tree(
            'engineered',
            self.engineered_data['X_train'],
            self.engineered_data['y_train']
        )
        self.trained_models['decision_tree_engineered'] = dt_engineered
        print("✅ Model 2/4 complete: Decision Tree (Engineered)\n")

        # 3. Random Forest - Original
        rf_original = self.train_random_forest(
            'original',
            self.original_data['X_train'],
            self.original_data['y_train']
        )
        self.trained_models['random_forest_original'] = rf_original
        print("✅ Model 3/4 complete: Random Forest (Original)\n")

        # 4. Random Forest - Engineered
        rf_engineered = self.train_random_forest(
            'engineered',
            self.engineered_data['X_train'],
            self.engineered_data['y_train']
        )
        self.trained_models['random_forest_engineered'] = rf_engineered
        print("✅ Model 4/4 complete: Random Forest (Engineered)\n")

        print("✅ All 4 models trained successfully!")

    def save_models(self, output_dir="phase_4_modeling/models/"):
        """Save all trained models and feature names for Phase 5."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n💾 SAVING MODELS TO {output_dir}")
        print("=" * 40)

        # Save all 4 models
        for model_name, model in self.trained_models.items():
            model_path = f"{output_dir}{model_name}.joblib"
            joblib.dump(model, model_path)
            print(f"   ✅ {model_name}.joblib")

        # Save feature names for both datasets
        if self.original_data:
            features_original = self.original_data['features']
            joblib.dump(features_original, f"{output_dir}feature_names_original.joblib")
            print(f"   ✅ feature_names_original.joblib ({len(features_original)} features)")

        if self.engineered_data:
            features_engineered = self.engineered_data['features']
            joblib.dump(features_engineered, f"{output_dir}feature_names_engineered.joblib")
            print(f"   ✅ feature_names_engineered.joblib ({len(features_engineered)} features)")

        print(f"\n📦 Total files saved: {len(self.trained_models) + 2}")
        print("🎯 Ready for Phase 5 evaluation!")

    def generate_training_report(self):
        """Generate Phase 4 training summary report."""
        print("\n" + "=" * 60)
        print("📋 PHASE 4 MODEL TRAINING SUMMARY")
        print("=" * 60)
        print("Team: Goblins Hiding in Vents")
        print("Phase: 4 - Model Development and Training")
        print("Team Leaders: Juan (ML Engineer), Julian (Feature Engineering)")
        print("=" * 60)

        print(f"\n🎯 TRAINING OBJECTIVES:")
        print(f"   • Train models on original (13) and engineered (20) features")
        print(f"   • Demonstrate value of Phase 3 feature engineering")
        print(f"   • Optimize hyperparameters with GridSearchCV")
        print(f"   • Use 10-fold stratified cross-validation")
        print(f"   • Target: ≥{self.target_accuracy:.1%} accuracy")
        print(f"   • Baseline: {self.baseline_accuracy:.1%} (hospital)")

        print(f"\n🤖 MODELS TRAINED:")
        print(f"   ✅ Decision Tree - Original (13 features)")
        print(f"   ✅ Decision Tree - Engineered (20 features)")
        print(f"   ✅ Random Forest - Original (13 features)")
        print(f"   ✅ Random Forest - Engineered (20 features)")

        print(f"\n📊 DATASETS USED:")
        if self.original_data:
            print(f"   Original: {len(self.original_data['X_train']) + len(self.original_data['X_test'])} samples")
            print(f"     - Train: {len(self.original_data['X_train'])}")
            print(f"     - Test: {len(self.original_data['X_test'])}")
            print(f"     - Features: 13 (age, anaemia, CPK, diabetes, EF, BP, etc.)")

        if self.engineered_data:
            print(f"\n   Engineered: {len(self.engineered_data['X_train']) + len(self.engineered_data['X_test'])} samples")
            print(f"     - Train: {len(self.engineered_data['X_train'])}")
            print(f"     - Test: {len(self.engineered_data['X_test'])}")
            print(f"     - Features: 20 (13 original + 7 engineered)")
            print(f"     - Engineered: kidney_heart_risk, cv_risk_score, severe_ef,")
            print(f"                   high_creatinine, low_sodium, age_time_risk, critical_patient")

        print(f"\n🔬 METHODOLOGY:")
        print(f"   • Hyperparameter optimization: GridSearchCV")
        print(f"   • Cross-validation: 10-fold stratified")
        print(f"   • Random state: 42 (reproducibility)")
        print(f"   • Scoring metric: Accuracy")

        print(f"\n📈 NEXT PHASE: PHASE 5 - MODEL EVALUATION")
        print(f"   Phase 5 will perform comprehensive evaluation:")
        print(f"   • ROC curve analysis and AUC scores")
        print(f"   • Bootstrap validation (confidence intervals)")
        print(f"   • Statistical significance testing vs baseline")
        print(f"   • Confusion matrices and detailed metrics")
        print(f"   • Compare original vs engineered feature performance")
        print(f"   • Select best overall model")

        print("\n" + "=" * 60)
        print("✅ PHASE 4 COMPLETE - MODELS READY FOR EVALUATION")
        print("=" * 60)


def main():
    """
    Main function to execute Phase 4 model training workflow.

    Trains 4 models (Decision Tree and Random Forest on both datasets)
    and saves them for Phase 5 evaluation.
    """
    print("=" * 60)
    print("PHASE 4: MODEL DEVELOPMENT & TRAINING")
    print("Team Leaders: Juan (ML Engineer), Julian (Feature Engineering)")
    print("Goal: Train models on original and engineered features")
    print("=" * 60)

    # Initialize trainer (loads both datasets)
    trainer = HeartFailureModelTrainer()

    if trainer.original_data and trainer.engineered_data:
        # Train all 4 models
        trainer.train_all_models()

        # Save models for Phase 5
        trainer.save_models()

        # Generate summary report
        trainer.generate_training_report()

    else:
        print("❌ Cannot proceed without both datasets. Check Phase 3 outputs.")


if __name__ == "__main__":
    main()
