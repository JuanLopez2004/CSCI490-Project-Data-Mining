# Simplified 3-Week Heart Failure Prediction

**Goal:** Beat hospital accuracy of 83.9%  
**Team:** 6 people, 2 per week  
**Dataset:** 1000 heart failure patients

## Week 1: Data Analysis (2 days)
**Team:** Jose + Julian

**Tasks:**
- Load CSV file and check size
- Calculate survival rate
- Find top 3 features that predict death
- Make correlation heatmap

**Code:** Use SIMPLE_WEEK_1.md
**Deliverable:** Top 3 feature names and correlation chart

## Week 2: Build Model (3 days)
**Team:** Juan + Nathan

**Tasks:**
- Split data into training and test sets
- Train Decision Tree, Random Forest, Logistic Regression
- Pick model with best accuracy
- Save best model as file

**Code:** Use SIMPLE_WEEK_2.md
**Deliverable:** Saved model file and accuracy score

## Week 3: Test and Present (2 days)
**Team:** Nasiru + Jose

**Tasks:**
- Load saved model and test it
- Create ROC curve and confusion matrix  
- Compare final accuracy to 83.9% baseline
- Write simple summary of results

**Code:** Use SIMPLE_WEEK_3.md
**Deliverable:** Results chart and final accuracy

## Key Simplifications Made

**Removed:**
- Complex statistical tests
- Advanced feature engineering
- Hyperparameter tuning
- Synthetic data generation
- Bootstrap validation
- Complex documentation

**Kept:**
- Basic data loading and analysis
- 3 simple machine learning models
- Model comparison and selection
- ROC curve and confusion matrix
- Clear success/failure vs baseline

## Success Definition

**Success:** Any model that beats 83.9% hospital baseline
**Documentation:** Simple chart showing our accuracy vs baseline
**Time:** Complete in 7 days total (2+3+2)