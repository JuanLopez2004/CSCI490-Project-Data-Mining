# Week 2 Plan: Machine Learning Model Development

**Team:** Juan and Nathan  
**Duration:** 7 days  
**Goal:** Build models that predict mortality in heart failure patients and outperform Seattle Heart Failure Model

**What you're predicting:** DEATH_EVENT (0 = patient survived, 1 = patient died from heart failure)

**Baseline to beat:** 
- Seattle Heart Failure Model: AUC-ROC ≈ 0.73 (~73% accuracy) (Levy et al., 2006)
- Your target: AUC-ROC ≥ 0.85 (≥85% accuracy)
- Goal: Statistically significant improvement with 95% confidence (bootstrapping)

## Day 1-2: Data Preparation and Splitting - Juan 

**What to code:**
- Load clean dataset from Week 1
- Separate features (X = all columns except DEATH_EVENT) from target (y = DEATH_EVENT)
- Split data using stratify=y to keep same death rate in both sets
- Verify death rate is ~32% in both training and test sets

**Expected output:**
- Training set with ~800 patients (80%)
- Test set with ~200 patients (20%)
- Both sets have same ~32% death rate
- X = 12 features, y = DEATH_EVENT (0 or 1)

## Day 3-4: Model Training

**What to code:**
- Train Decision Tree with class_weight='balanced'
- Train Random Forest with class_weight='balanced'
- Train Logistic Regression with class_weight='balanced'
- Set random_state=42 for reproducible results

**Expected output:**
- 3 trained models that predict death (1) vs survival (0)
- Models ready to test on 200 test patients
- Each model trained on ~800 training patients

## Day 5-6: Model Evaluation and Comparison

**What to code:**
- Test all 3 models on test set (predict death vs survival)
- Calculate accuracy and AUC-ROC for each model
- Compare to Seattle HF Model (AUC ≈ 0.73)
- Generate classification reports (precision/recall/F1-score)
- Create confusion matrices
- Calculate ROC curves

**Expected output:**
- Accuracy: Does it reach ≥85% target?
- AUC-ROC: Does it beat 0.73 baseline and reach ≥0.85?
- Precision: When model predicts death, how often is it right?
- Recall/Sensitivity: Of patients who died, what % did model catch?
- F1-score: Balance between precision and recall
- ROC curve: Visual comparison to Seattle model

## Day 7: Best Model Selection and Saving - Juan

**What to code:**
- Select model with highest AUC-ROC (target: ≥0.85)
- Calculate feature importance
- Perform bootstrap validation (1000 iterations) for 95% confidence intervals
- Save best model as best_model.pkl
- Document: accuracy, AUC-ROC, confidence intervals, top features

**Deliverables for Week 3 team:**
- best_model.pkl (can predict death for new patients)
- Final accuracy score (target: ≥85%)
- AUC-ROC score (target: ≥0.85, beating Seattle's 0.73)
- 95% confidence intervals from bootstrapping
- Feature importance: which of the 12 features predict death best
- Summary: "Our [model] achieved AUC-ROC of X vs Seattle HF Model (0.73)"

**Success criteria:**
- Model achieves AUC-ROC ≥0.85 (beats Seattle's 0.73)
- Accuracy ≥85%
- 95% confidence intervals show statistically significant improvement
- Feature importance shows top predictors (time, ejection_fraction, serum_creatinine)
- Clear handoff to Week 3 with working model file