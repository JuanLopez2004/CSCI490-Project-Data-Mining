# Phase 4: Model Development and Training

## Goal
Train Decision Tree and Random Forest models on both original and engineered datasets to demonstrate the value of Phase 3's feature engineering.

## Team
Nathan and Juan lead this phase.

## What to do
1. Load original dataset (13 features) from Phase 3
2. Load engineered dataset (20 features) from Phase 3
3. Train Decision Tree and Random Forest on both datasets (4 models total)
4. Optimize hyperparameters with GridSearchCV
5. Save all 4 models for Phase 5 evaluation

## Models Trained
- **Decision Tree (Original)**: 13 features from Phase 3 split data
- **Decision Tree (Engineered)**: 20 features (13 original + 7 engineered)
- **Random Forest (Original)**: 13 features from Phase 3 split data
- **Random Forest (Engineered)**: 20 features (13 original + 7 engineered)

## Implementation Details
- **Hyperparameter Tuning**: GridSearchCV tests hundreds of parameter combinations
- **Cross-Validation**: 10-fold stratified CV for robust evaluation
- **Progress Tracking**: Real-time progress bars show training status
- **Parallelization**: CV folds run in parallel for faster training
- **Data Sources**:
  - Original: `phase_3_features/datasets/train_data.csv` + `test_data.csv`
  - Engineered: `phase_3_features/results/data_with_engineered_features.csv`

## Running Phase 4
```bash
# From csci490/ directory
python phase_4_modeling/model_trainer.py
```

Expected output:
- Training progress for all 4 models with progress bars
- Best hyperparameters found for each model
- Cross-validation accuracy scores
- All models saved to `phase_4_modeling/models/`

## Outputs Saved for Phase 5
All files saved to `phase_4_modeling/models/`:
- `decision_tree_original.joblib` - Decision Tree trained on 13 features
- `decision_tree_engineered.joblib` - Decision Tree trained on 20 features
- `random_forest_original.joblib` - Random Forest trained on 13 features
- `random_forest_engineered.joblib` - Random Forest trained on 20 features
- `feature_names_original.joblib` - List of 13 original feature names
- `feature_names_engineered.joblib` - List of 20 engineered feature names

## Success Criteria
- ✅ All 4 models trained successfully with GridSearchCV
- ✅ Hyperparameter optimization completes for each model
- ✅ Cross-validation scores calculated (10-fold stratified)
- ✅ All 6 files saved to models directory
- ✅ Ready for Phase 5 comprehensive evaluation

## Next Phase
**Phase 5: Model Evaluation & Validation**
- Load all 4 trained models
- Generate ROC curves and AUC analysis
- Perform bootstrap validation for confidence intervals
- Statistical significance testing vs 83.9% baseline
- Compare original vs engineered feature performance
- Select best overall model