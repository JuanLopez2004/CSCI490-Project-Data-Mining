"""
Phase 3: Feature Engineering & Selection
Heart Failure Mortality Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("PHASE 3: FEATURE ENGINEERING & SELECTION")
print("=" * 70)

# ================================================
# 1. LOAD DATA
# ================================================
print("\n1. Loading data...")
df = pd.read_csv('../Datasets/training_data.csv')

print(f"   Total patients: {len(df)}")
print(f"   Features: {len(df.columns) - 1}")
print(f"   Death rate: {df['DEATH_EVENT'].mean():.2%}")

X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# ================================================
# 2. CORRELATION ANALYSIS
# ================================================
print("\n2. Analyzing correlations with mortality...")
correlations = df.corr()['DEATH_EVENT'].drop('DEATH_EVENT').sort_values(ascending=False)

print("\n   Feature Correlations with Death:")
print("   " + "=" * 50)
for feature, corr in correlations.items():
    print(f"   {feature:30s}: {corr:+.4f}")

# Visualize
plt.figure(figsize=(10, 8))
correlations.plot(kind='barh', color=['green' if x < 0 else 'red' for x in correlations])
plt.xlabel('Correlation with Death')
plt.title('Feature Correlations with Mortality', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: feature_correlations.png")

# ================================================
# 3. STATISTICAL SIGNIFICANCE
# ================================================
print("\n3. Testing statistical significance...")
p_values = {}

for column in X.columns:
    survived = df[df['DEATH_EVENT'] == 0][column]
    died = df[df['DEATH_EVENT'] == 1][column]
    t_stat, p_val = stats.ttest_ind(survived, died)
    p_values[column] = p_val

significance_df = pd.DataFrame({
    'Feature': correlations.index,
    'Correlation': correlations.values,
    'P-Value': [p_values[f] for f in correlations.index],
    'Significant': [p_values[f] < 0.05 for f in correlations.index]
})

print("\n   Statistical Significance (α = 0.05):")
print("   " + "=" * 65)
print(f"   {'Feature':<30} {'Correlation':>12} {'P-Value':>12} {'Sig':>8}")
print("   " + "=" * 65)

for _, row in significance_df.iterrows():
    sig_mark = "✓" if row['Significant'] else "✗"
    print(f"   {row['Feature']:<30} {row['Correlation']:>+12.4f} {row['P-Value']:>12.6f} {sig_mark:>8}")

significant_features = significance_df[significance_df['Significant']]['Feature'].tolist()
print(f"\n   Significant features (p < 0.05): {len(significant_features)}")

# ================================================
# 4. MUTUAL INFORMATION
# ================================================
print("\n4. Calculating mutual information scores...")
mi_scores = mutual_info_classif(X, y, random_state=42)

mi_df = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print("\n   Mutual Information Scores:")
print("   " + "=" * 50)
for _, row in mi_df.iterrows():
    print(f"   {row['Feature']:<30} {row['MI_Score']:.4f}")

# Visualize
plt.figure(figsize=(10, 8))
plt.barh(mi_df['Feature'], mi_df['MI_Score'], color='steelblue')
plt.xlabel('Mutual Information Score')
plt.title('Feature Importance: Mutual Information', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mutual_information_scores.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: mutual_information_scores.png")

# ================================================
# 5. RANDOM FOREST IMPORTANCE
# ================================================
print("\n5. Training Random Forest for feature importance...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X, y)

rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n   Random Forest Feature Importance:")
print("   " + "=" * 50)
for _, row in rf_importance.iterrows():
    print(f"   {row['Feature']:<30} {row['Importance']:.4f}")

# Visualize
plt.figure(figsize=(10, 8))
plt.barh(rf_importance['Feature'], rf_importance['Importance'], color='forestgreen')
plt.xlabel('Importance')
plt.title('Feature Importance: Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: rf_feature_importance.png")

# ================================================
# 6. FEATURE ENGINEERING
# ================================================
print("\n6. Engineering new features...")
df_engineered = df.copy()

# Create interaction features
df_engineered['kidney_heart_risk'] = df['serum_creatinine'] * (100 - df['ejection_fraction'])
df_engineered['cv_risk_score'] = df['high_blood_pressure'] + df['diabetes'] + df['smoking'] + df['anaemia']
df_engineered['severe_ef'] = (df['ejection_fraction'] < 30).astype(int)
df_engineered['high_creatinine'] = (df['serum_creatinine'] > 1.5).astype(int)
df_engineered['low_sodium'] = (df['serum_sodium'] < 135).astype(int)
df_engineered['age_time_risk'] = df['age'] / (df['time'] + 1)
df_engineered['critical_patient'] = df_engineered['severe_ef'] * df_engineered['high_creatinine']

new_features = ['kidney_heart_risk', 'cv_risk_score', 'severe_ef', 'high_creatinine', 
                'low_sodium', 'age_time_risk', 'critical_patient']

print(f"\n   Created {len(new_features)} engineered features:")
for feature in new_features:
    print(f"     - {feature}")

# Evaluate engineered features
engineered_corr = df_engineered[new_features + ['DEATH_EVENT']].corr()['DEATH_EVENT'].drop('DEATH_EVENT').sort_values(ascending=False)

print("\n   Engineered Feature Correlations:")
print("   " + "=" * 50)
for feature, corr in engineered_corr.items():
    print(f"   {feature:<30} {corr:+.4f}")

# ================================================
# 7. FEATURE SELECTION RECOMMENDATION
# ================================================
print("\n7. Generating feature recommendations...")

feature_rankings = pd.DataFrame({
    'Feature': X.columns,
    'Correlation_Rank': [list(correlations.index).index(f) + 1 for f in X.columns],
    'MI_Rank': [list(mi_df['Feature']).index(f) + 1 for f in X.columns],
    'RF_Rank': [list(rf_importance['Feature']).index(f) + 1 for f in X.columns],
    'Significant': [f in significant_features for f in X.columns]
})

feature_rankings['Avg_Rank'] = feature_rankings[['Correlation_Rank', 'MI_Rank', 'RF_Rank']].mean(axis=1)
feature_rankings = feature_rankings.sort_values('Avg_Rank')

print("\n   Feature Ranking Summary:")
print("   " + "=" * 85)
print(f"   {'Feature':<30} {'Corr':>6} {'MI':>6} {'RF':>6} {'Avg':>7} {'Sig':>5}")
print("   " + "=" * 85)

for _, row in feature_rankings.iterrows():
    sig_mark = "✓" if row['Significant'] else "✗"
    print(f"   {row['Feature']:<30} {row['Correlation_Rank']:>6.0f} {row['MI_Rank']:>6.0f} "
          f"{row['RF_Rank']:>6.0f} {row['Avg_Rank']:>7.2f} {sig_mark:>5}")

top_features = feature_rankings.head(5)['Feature'].tolist()

print("\n" + "=" * 85)
print("\n🎯 RECOMMENDED FEATURE SET (Top 5):")
print("=" * 85)
for i, feature in enumerate(top_features, 1):
    corr = correlations[feature]
    mi = mi_df[mi_df['Feature'] == feature]['MI_Score'].values[0]
    rf_imp = rf_importance[rf_importance['Feature'] == feature]['Importance'].values[0]
    print(f"   {i}. {feature}")
    print(f"      - Correlation: {corr:+.4f}")
    print(f"      - Mutual Info: {mi:.4f}")
    print(f"      - RF Importance: {rf_imp:.4f}")
    print()

# ================================================
# 8. VISUALIZE TOP FEATURES
# ================================================
print("8. Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(top_features):
    ax = axes[idx]
    df.boxplot(column=feature, by='DEATH_EVENT', ax=ax)
    ax.set_title(f'{feature}\n(corr: {correlations[feature]:+.3f})')
    ax.set_xlabel('Death Event (0=Survived, 1=Died)')
    ax.set_ylabel(feature)
    plt.sca(ax)
    plt.xticks([1, 2], ['Survived', 'Died'])

fig.delaxes(axes[5])
plt.suptitle('Top 5 Predictive Features by Mortality Outcome', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('top_features_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: top_features_distribution.png")

# ================================================
# 9. SAVE RESULTS
# ================================================
print("\n9. Saving results...")

# Save feature rankings
feature_rankings.to_csv('feature_importance_summary.csv', index=False)
print("   ✓ Saved: feature_importance_summary.csv")

# Save top features
with open('top_features.txt', 'w') as f:
    f.write("Top 5 Features for Week 2 Modeling:\n")
    f.write("=" * 70 + "\n\n")
    for i, feature in enumerate(top_features, 1):
        f.write(f"{i}. {feature}\n")
        f.write(f"   - Correlation: {correlations[feature]:+.4f}\n")
        f.write(f"   - Mutual Info: {mi_df[mi_df['Feature'] == feature]['MI_Score'].values[0]:.4f}\n")
        f.write(f"   - RF Importance: {rf_importance[rf_importance['Feature'] == feature]['Importance'].values[0]:.4f}\n\n")
    
    f.write("\nThese features consistently ranked highest across:\n")
    f.write("  - Correlation analysis\n")
    f.write("  - Mutual information\n")
    f.write("  - Random Forest importance\n")
    f.write("  - Statistical significance (p < 0.05)\n")

print("   ✓ Saved: top_features.txt")

# Save engineered dataset
df_engineered.to_csv('data_with_engineered_features.csv', index=False)
print("   ✓ Saved: data_with_engineered_features.csv")

# ================================================
# FINAL SUMMARY
# ================================================
print("\n" + "=" * 70)
print("PHASE 3 COMPLETE!")
print("=" * 70)

print("\n📊 KEY FINDINGS:")
print(f"   - Total features analyzed: {len(X.columns)}")
print(f"   - Statistically significant features: {len(significant_features)}")
print(f"   - Engineered features created: {len(new_features)}")
print(f"   - Recommended features for modeling: {len(top_features)}")

print("\n🎯 TOP 5 FEATURES:")
for i, feature in enumerate(top_features, 1):
    print(f"   {i}. {feature} (correlation: {correlations[feature]:+.4f})")

print("\n📁 DELIVERABLES:")
print("   1. feature_correlations.png")
print("   2. mutual_information_scores.png")
print("   3. rf_feature_importance.png")
print("   4. top_features_distribution.png")
print("   5. feature_importance_summary.csv")
print("   6. top_features.txt")
print("   7. data_with_engineered_features.csv")

print("\n✓ Week 2 team (Juan & Nathan) should use these top 5 features for modeling")
print("=" * 70)
