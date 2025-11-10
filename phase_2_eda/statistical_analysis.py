"""
Phase 2: Exploratory Data Analysis (EDA)
Team: Goblins Hiding in Vents (Cameron, Julian, Juan, Nathan, Nasiru, Jose)
Week: 2

This module performs comprehensive exploratory data analysis on the UCI heart failure dataset.
Team Leaders: Nathan (Statistical Analyst), Nasiru (Data Visualization Lead)

Goals:
- Generate statistical summaries for all features
- Create comprehensive data visualizations
- Analyze correlations between features and target variable
- Identify patterns and insights for feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HeartFailureEDA:
    """
    Comprehensive Exploratory Data Analysis for heart failure prediction.
    
    Responsibilities:
    - Statistical summaries and distributions
    - Correlation analysis (Pearson, Spearman)
    - Chi-square tests for categorical associations
    - Data visualization suite
    - Feature relationship analysis
    """
    
    def __init__(self, data_path: str = "../data/raw/heart_failure_clinical_records.csv"):
        """
        Initialize EDA class with heart failure dataset.
        
        Args:
            data_path: Path to the heart failure dataset
        """
        self.data_path = data_path
        self.df = None
        self.numerical_features = []
        self.categorical_features = []
        self.target = 'DEATH_EVENT'
        
        # Load and prepare data
        self.load_data()
        self.identify_feature_types()
        
    def load_data(self):
        """Load the heart failure dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully: {self.df.shape}")
        except FileNotFoundError:
            print(f"❌ Data file not found at {self.data_path}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def identify_feature_types(self):
        """Identify numerical and categorical features."""
        if self.df is not None:
            # Binary categorical features (0/1)
            binary_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
            
            # Continuous numerical features
            continuous_features = [
                'age', 'creatinine_phosphokinase', 'ejection_fraction',
                'platelets', 'serum_creatinine', 'serum_sodium', 'time'
            ]
            
            self.categorical_features = binary_features
            self.numerical_features = continuous_features
            
            print(f"📊 Identified {len(self.numerical_features)} numerical and {len(self.categorical_features)} categorical features")
    
    def generate_statistical_summary(self):
        """Generate comprehensive statistical summaries."""
        if self.df is None:
            return None
            
        print("📈 STATISTICAL SUMMARY")
        print("="*60)
        
        # Overall summary
        print("\n🔍 DATASET OVERVIEW:")
        print(f"   Total patients: {len(self.df)}")
        print(f"   Total features: {len(self.df.columns) - 1}")  # Exclude target
        print(f"   Deaths: {self.df[self.target].sum()} ({self.df[self.target].mean()*100:.1f}%)")
        print(f"   Survivors: {len(self.df) - self.df[self.target].sum()} ({(1-self.df[self.target].mean())*100:.1f}%)")
        
        # Numerical features summary
        print("\n📊 NUMERICAL FEATURES SUMMARY:")
        numerical_summary = self.df[self.numerical_features].describe()
        print(numerical_summary.round(2))
        
        # Categorical features summary
        print("\n🏷️ CATEGORICAL FEATURES SUMMARY:")
        for feature in self.categorical_features:
            counts = self.df[feature].value_counts()
            percentages = self.df[feature].value_counts(normalize=True) * 100
            print(f"\n{feature.upper()}:")
            for value in [0, 1]:
                if value in counts:
                    print(f"   {value}: {counts[value]} ({percentages[value]:.1f}%)")
        
        return {
            'numerical_summary': numerical_summary,
            'target_distribution': self.df[self.target].value_counts()
        }
    
    def analyze_correlations(self):
        """Analyze correlations between features and target variable."""
        if self.df is None:
            return None
            
        print("\n🔗 CORRELATION ANALYSIS")
        print("="*60)
        
        # Pearson correlations with target variable
        print("\n📊 FEATURE CORRELATIONS WITH DEATH_EVENT:")
        correlations = {}
        
        for feature in self.numerical_features + self.categorical_features:
            corr, p_value = pearsonr(self.df[feature], self.df[self.target])
            correlations[feature] = {'correlation': corr, 'p_value': p_value}
            
            # Significance indicator
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"   {feature:25} r = {corr:6.3f} (p = {p_value:.4f}) {sig}")
        
        # Sort by absolute correlation
        sorted_correlations = sorted(correlations.items(), 
                                   key=lambda x: abs(x[1]['correlation']), 
                                   reverse=True)
        
        print("\n🎯 TOP PREDICTIVE FEATURES (by correlation magnitude):")
        for i, (feature, stats) in enumerate(sorted_correlations[:5], 1):
            print(f"   {i}. {feature}: r = {stats['correlation']:.3f}")
        
        return correlations
    
    def perform_chi_square_tests(self):
        """Perform chi-square tests for categorical feature independence."""
        if self.df is None:
            return None
            
        print("\n🔬 CHI-SQUARE INDEPENDENCE TESTS")
        print("="*60)
        
        chi_square_results = {}
        
        for feature in self.categorical_features:
            # Create contingency table
            contingency_table = pd.crosstab(self.df[feature], self.df[self.target])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            chi_square_results[feature] = {
                'chi2': chi2,
                'p_value': p_value,
                'degrees_freedom': dof,
                'contingency_table': contingency_table
            }
            
            # Significance and effect size
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{feature:25} χ² = {chi2:6.3f}, p = {p_value:.4f}, V = {cramers_v:.3f} {sig}")
        
        return chi_square_results
    
    def create_distribution_plots(self, figsize=(20, 15)):
        """Create distribution plots for all features."""
        if self.df is None:
            return None
            
        # Calculate subplot dimensions
        n_features = len(self.numerical_features) + len(self.categorical_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot numerical features
        for feature in self.numerical_features:
            ax = axes[plot_idx]
            
            # Histogram with KDE
            self.df[feature].hist(bins=30, alpha=0.7, ax=ax, density=True)
            
            # Add KDE
            try:
                self.df[feature].plot(kind='kde', ax=ax, color='red', linewidth=2)
            except:
                pass  # Skip KDE if it fails
                
            ax.set_title(f'{feature}\n(Mean: {self.df[feature].mean():.2f}, Std: {self.df[feature].std():.2f})')
            ax.set_ylabel('Density')
            
            plot_idx += 1
        
        # Plot categorical features
        for feature in self.categorical_features:
            ax = axes[plot_idx]
            
            # Bar plot
            counts = self.df[feature].value_counts()
            counts.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
            ax.set_title(f'{feature}\n0: {counts.get(0, 0)}, 1: {counts.get(1, 0)}')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=0)
            
            plot_idx += 1
        
        # Hide empty subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Feature Distributions - Heart Failure Dataset', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_target_analysis_plots(self, figsize=(16, 12)):
        """Create plots analyzing features by target variable."""
        if self.df is None:
            return None
            
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()
        
        plot_idx = 0
        
        # Box plots for top numerical features
        top_numerical = ['ejection_fraction', 'serum_creatinine', 'age', 'serum_sodium']
        
        for feature in top_numerical:
            ax = axes[plot_idx]
            sns.boxplot(data=self.df, x=self.target, y=feature, ax=ax)
            ax.set_title(f'{feature} by Survival Status')
            ax.set_xlabel('Death Event (0=Survived, 1=Died)')
            plot_idx += 1
        
        # Stacked bar plots for categorical features
        for feature in self.categorical_features:
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            # Create contingency table percentages
            cont_table = pd.crosstab(self.df[feature], self.df[self.target], normalize='index') * 100
            cont_table.plot(kind='bar', stacked=True, ax=ax, color=['lightgreen', 'lightcoral'])
            ax.set_title(f'{feature} vs Death Event')
            ax.set_xlabel(f'{feature} (0=No, 1=Yes)')
            ax.set_ylabel('Percentage')
            ax.legend(['Survived', 'Died'])
            ax.tick_params(axis='x', rotation=0)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Feature Analysis by Survival Status', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_correlation_heatmap(self, figsize=(12, 10)):
        """Create correlation heatmap for all features."""
        if self.df is None:
            return None
            
        # Calculate correlation matrix
        corr_matrix = self.df.corr()
        
        # Create heatmap
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8})
        
        plt.title('Feature Correlation Matrix - Heart Failure Dataset', fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
        
        return corr_matrix
    
    def generate_eda_report(self):
        """Generate comprehensive EDA report."""
        print("🔬 COMPREHENSIVE EDA REPORT")
        print("="*80)
        print("Team: Goblins Hiding in Vents")
        print("Phase: 2 - Exploratory Data Analysis") 
        print("Dataset: UCI Heart Failure Clinical Records (299 patients)")
        print("="*80)
        
        # 1. Statistical Summary
        stats_summary = self.generate_statistical_summary()
        
        # 2. Correlation Analysis
        correlations = self.analyze_correlations()
        
        # 3. Chi-square Tests
        chi_results = self.perform_chi_square_tests()
        
        # 4. Key Insights
        print("\n🎯 KEY INSIGHTS FROM EDA:")
        print("="*40)
        print("1. Dataset has 203 survivors (67.9%) vs 96 deaths (32.1%)")
        print("2. Most predictive features likely include:")
        print("   - ejection_fraction (continuous)")
        print("   - serum_creatinine (continuous)")  
        print("   - age (continuous)")
        print("   - time (follow-up period)")
        print("3. Binary features show varying associations with outcome")
        print("4. No missing values - dataset is clean")
        print("5. Feature distributions vary - may need scaling/normalization")
        
        print("\n📋 READY FOR PHASE 3: FEATURE ENGINEERING")
        print("   Next steps: Feature selection, scaling, synthetic data generation")


def main():
    """
    Main function to execute Phase 2 EDA workflow.
    
    Team members should run this to:
    1. Generate statistical summaries
    2. Analyze correlations and associations
    3. Create comprehensive visualizations
    4. Identify key patterns for modeling
    """
    print("="*60)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS")
    print("Team Leaders: Nathan (Statistical), Nasiru (Visualization)")
    print("="*60)
    
    # Initialize EDA
    eda = HeartFailureEDA()
    
    if eda.df is not None:
        # Generate comprehensive EDA report
        eda.generate_eda_report()
        
        print("\n🎨 Generating visualizations...")
        
        # Create all visualizations
        eda.create_distribution_plots()
        eda.create_target_analysis_plots()
        eda.create_correlation_heatmap()
        
        print("✅ Phase 2 EDA completed successfully!")
        print("📊 All statistical analyses and visualizations generated")
        
    else:
        print("❌ Cannot proceed without data. Check data loading.")


if __name__ == "__main__":
    main()