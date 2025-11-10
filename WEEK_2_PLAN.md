# Week 2 Plan: Machine Learning Model Development

**Team:** Juan and Nathan  
**Duration:** 7 days  
**Goal:** Build models that beat hospital baseline of 83.9% accuracy

## Day 1-2: Data Preparation and Splitting

**What to code:**
- Load clean dataset from Week 1
- Separate features (X) from target variable (y)
- Split data into training and testing sets (80/20 split)
- Verify data splitting maintains death rate balance

**Expected output:**
- Training set with ~239 patients
- Test set with ~60 patients
- Same death rate in both training and test sets
- Features and target properly separated

## Day 3-4: Model Training

**What to code:**
- Train Decision Tree Classifier
- Train Random Forest Classifier
- Train Logistic Regression
- Use cross-validation to estimate performance
- Set random seeds for reproducible results

**Expected output:**
- 3 trained models ready for testing
- Cross-validation scores for each model
- Models that can make predictions on new data

## Day 5-6: Model Evaluation and Comparison

**What to code:**
- Test all 3 models on test set
- Calculate accuracy for each model
- Compare each accuracy to 83.9% hospital baseline
- Generate classification reports
- Create confusion matrices

**Expected output:**
- Accuracy scores for all 3 models
- Clear identification of which models beat baseline
- Detailed performance metrics (precision, recall, F1-score)
- Confusion matrices showing prediction errors

## Day 7: Best Model Selection and Saving

**What to code:**
- Select model with highest accuracy
- Calculate feature importance (if available)
- Save best model as pickle file
- Document final performance vs baseline

**Deliverables for Week 3 team:**
- Saved best model file (best_model.pkl)
- Final accuracy score
- Feature importance rankings
- Performance comparison to hospital baseline

**Success criteria:**
- At least one model beats 83.9% baseline
- Best model saved and ready for testing
- Clear documentation of which model performs best
- Feature importance analysis completed