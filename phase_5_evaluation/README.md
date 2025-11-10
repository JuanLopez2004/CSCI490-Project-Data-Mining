# Week 3: Model Testing and Results

## Goal
Test the saved model and create professional results presentation.

## Team
Nasiru and Jose work on this week.

## What to do
1. Load the best model from Week 2
2. Test it thoroughly with ROC curves and confusion matrix
3. Calculate final accuracy vs hospital baseline
4. Create simple presentation showing results

## Key tests
- **ROC Curve**: Shows how good the model is at distinguishing outcomes
- **Confusion Matrix**: Shows what the model got right vs wrong
- **Accuracy**: Final percentage vs 83.9% hospital baseline

## Simple testing approach
```python
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import pickle

# Load saved model
model = pickle.load(open('best_model.pkl', 'rb'))

# Test predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f'Model (AUC = {auc:.3f})')
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve')
plt.legend()
plt.show()

# Final results
accuracy = accuracy_score(y_test, y_pred)
print(f'Our model: {accuracy:.3f}')
print(f'Hospital baseline: 0.839')
print(f'Improvement: {accuracy - 0.839:.3f}')
```

## Results to show
- Final accuracy percentage
- How much we beat hospital baseline by
- ROC curve showing model performance
- Confusion matrix showing correct vs incorrect predictions

## Success check
- Model tests run without errors
- Final accuracy calculated
- ROC curve and confusion matrix created
- Clear comparison to 83.9% baseline
- Simple presentation prepared

## Files to create
- `week3_evaluation.py` - Main testing script
- `final_results.txt` - Summary of performance
- `roc_curve.png` - ROC curve chart
- `confusion_matrix.png` - Confusion matrix chart

## Deliverables
- Working model that beats baseline
- Professional charts showing performance  
- Clear documentation of results
- Presentation ready for submission