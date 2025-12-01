# Week 2: Machine Learning Models

## Goal
Build and test 3 different models to predict heart failure.

## Team
Juan and Nathan work on this week.

## What to do
1. Split data into training and testing sets
2. Try 2 models: Decision Tree, Random Forest
3. Find which model has best accuracy
4. Save the best model

## Models to try
- **Decision Tree**: Easy to understand, shows decision rules
- **Random Forest**: Combines many decision trees

## Simple approach
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Find best model
for name, model in models.items():
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    print(f'{name}: {score:.3f}')
```

## Target
- Beat hospital baseline of 83.9% accuracy
- Save best performing model
- Document which features are most important

## Success check
- Both models trained successfully
- Best model accuracy > 0.839 (83.9%)
- Model saved as pickle file
- Know which model works best

## Files to create
- `week2_modeling.py` - Main training script
- `best_model.pkl` - Saved best model

## Next week
Move to Week 3: Test the saved model thoroughly and create final presentation.