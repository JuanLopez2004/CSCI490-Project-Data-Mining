# Week 1: Data Collection and Setup

## Goal
Load the UCI heart failure dataset and understand the basic structure.

## Team
Cameron and Julian work on this week.

## What to do
1. Load the dataset (299 patients)
2. Check data quality (missing values, data types)
3. Create basic summary statistics
4. Make simple charts to understand the data

## Files
- `week1_data_analysis.py` - Simple data loading and analysis
- `original.csv` - Heart failure dataset

## Dataset basics
- 299 patients
- 12 features: age, blood pressure, ejection fraction, etc.
- Target: DEATH_EVENT (0=survived, 1=died)
- No missing values

## Key features to focus on
- age: Patient age
- ejection_fraction: Heart pumping strength (most important)
- serum_creatinine: Kidney function
- time: Follow-up days
- DEATH_EVENT: What we want to predict

## Simple code example
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('original.csv')

# Basic info
print(f"Dataset size: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Target distribution
df['DEATH_EVENT'].value_counts().plot(kind='bar')
plt.title('Survival vs Death')
plt.show()
```

## Success check
- Data loads without errors
- 299 rows, 13 columns
- No missing values
- Basic charts created
- Understanding which patients died vs survived

## Next week
Move to Week 2: Build machine learning models