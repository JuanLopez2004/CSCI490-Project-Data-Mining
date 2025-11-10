# Week 1: Data Analysis (Extended)

## Goal
Understand the heart failure data through basic statistics and charts.

## Team
Cameron and Julian continue working together.

## What to do
1. Calculate basic statistics for all features
2. Create correlation heatmap
3. Make charts showing survival patterns
4. Identify which features matter most

## Key questions to answer
- Which features correlate with death events?
- What's the survival rate in the dataset?
- Are there obvious patterns in the data?
- Which patients are highest risk?

## Simple analysis steps
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('original.csv')

# Basic statistics
print(df.describe())

# Survival rate
survival_rate = 1 - df['DEATH_EVENT'].mean()
print(f"Survival rate: {survival_rate:.1%}")

# Correlation with death
correlations = df.corr()['DEATH_EVENT'].sort_values()
print("Features most correlated with death:")
print(correlations)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, center=0)
plt.title('Feature Correlations')
plt.show()
```

## Important findings to look for
- Ejection fraction: Low values = higher death risk
- Age: Older patients = higher death risk  
- Serum creatinine: High values = kidney problems = higher death risk
- Time: Longer follow-up = more complete data

## Charts to create
1. Survival vs death bar chart
2. Age distribution by outcome
3. Ejection fraction by outcome
4. Correlation heatmap

## Success check
- Calculated survival rate (~68%)
- Identified top 3 predictive features
- Created correlation heatmap
- Understand which patients are high risk

## Files created
- Basic statistics summary
- Correlation analysis results
- Simple charts showing patterns

## Next week
Move to Week 2: Build machine learning models using the insights from this analysis.