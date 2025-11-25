"""
Split dataset into training and test sets
80/20 split with stratification to maintain death rate
"""

import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 70)
print("SPLITTING DATA INTO TRAIN/TEST SETS")
print("=" * 70)

# Load data
df = pd.read_csv('../Datasets/training_data.csv')
print(f"\nTotal dataset: {len(df)} patients")
print(f"Death rate: {df['DEATH_EVENT'].mean():.2%}")

# Separate features and target
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Stratified split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\n✅ Data split complete:")
print(f"   Training set: {len(X_train)} patients ({len(X_train)/len(df)*100:.1f}%)")
print(f"   Test set: {len(X_test)} patients ({len(X_test)/len(df)*100:.1f}%)")

print(f"\n✅ Death rates maintained:")
print(f"   Training death rate: {y_train.mean():.2%}")
print(f"   Test death rate: {y_test.mean():.2%}")

# Combine X and y for saving
train_df = X_train.copy()
train_df['DEATH_EVENT'] = y_train

test_df = X_test.copy()
test_df['DEATH_EVENT'] = y_test

# Save splits
train_df.to_csv('../Datasets/train_data.csv', index=False)
test_df.to_csv('../Datasets/test_data.csv', index=False)

print(f"\n💾 Saved files:")
print(f"   ../Datasets/train_data.csv ({len(train_df)} rows)")
print(f"   ../Datasets/test_data.csv ({len(test_df)} rows)")
print(f"\n✅ Ready for Phase 4 modeling!")
