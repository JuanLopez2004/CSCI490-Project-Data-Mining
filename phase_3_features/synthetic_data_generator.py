"""
Phase 3: Synthetic Data Generation
Team: Goblins Hiding in Vents
Lead: Jose (Synthetic Data Generator)
Support: Julian (Feature Engineering Specialist)

This module generates 701 synthetic heart failure records while maintaining
the statistical distributions and correlations found in the original UCI dataset.
Goal: Expand from 299 real records to 1000 total records (299 + 701 synthetic).
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SyntheticDataGenerator:
    """
    Advanced synthetic data generation for heart failure dataset expansion.
    
    Responsibilities:
    - Analyze original data distributions (marginal and joint)
    - Generate 701 synthetic records maintaining statistical properties
    - Preserve feature correlations and clinical relationships
    - Validate synthetic data quality and medical realism
    - Create balanced dataset for improved ML model training
    """
    
    def __init__(self, data_path: str = "../data/raw/heart_failure_clinical_records.csv"):
        """
        Initialize synthetic data generator with original UCI dataset.
        
        Args:
            data_path: Path to original heart failure dataset
        """
        self.data_path = data_path
        self.original_df = None
        self.synthetic_df = None
        self.combined_df = None
        
        # Feature categories
        self.numerical_features = [
            'age', 'creatinine_phosphokinase', 'ejection_fraction',
            'platelets', 'serum_creatinine', 'serum_sodium', 'time'
        ]
        
        self.categorical_features = [
            'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking'
        ]
        
        self.target = 'DEATH_EVENT'
        
        # Statistical models
        self.distribution_models = {}
        self.correlation_matrix = None
        self.target_distributions = {}
        
        self.load_and_analyze_data()
    
    def load_and_analyze_data(self):
        """Load original data and analyze distributions."""
        try:
            self.original_df = pd.read_csv(self.data_path)
            print(f"✅ Original data loaded: {self.original_df.shape}")
            
            # Analyze distributions
            self.analyze_feature_distributions()
            self.analyze_correlations()
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def analyze_feature_distributions(self):
        """Analyze feature distributions for synthetic data generation."""
        if self.original_df is None:
            return None
            
        print("📊 ANALYZING FEATURE DISTRIBUTIONS")
        print("="*40)
        
        # Analyze by target class for stratified generation
        for target_value in [0, 1]:
            target_name = "Survived" if target_value == 0 else "Died"
            subset = self.original_df[self.original_df[self.target] == target_value]
            
            print(f"\n🎯 {target_name} Group (n={len(subset)}):")
            
            self.target_distributions[target_value] = {}
            
            # Numerical features - fit distributions
            for feature in self.numerical_features:
                data = subset[feature].values
                
                # Try different distributions and select best fit
                distributions = [stats.norm, stats.lognorm, stats.gamma, stats.beta]
                best_dist = None
                best_params = None
                best_sse = np.inf
                
                for dist in distributions:
                    try:
                        params = dist.fit(data)
                        # Calculate goodness of fit (Kolmogorov-Smirnov test)
                        ks_stat, p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))
                        
                        if ks_stat < best_sse:
                            best_sse = ks_stat
                            best_dist = dist
                            best_params = params
                    except:
                        continue
                
                self.target_distributions[target_value][feature] = {
                    'distribution': best_dist,
                    'params': best_params,
                    'original_data': data
                }
                
                print(f"     {feature:25} Best fit: {best_dist.name if best_dist else 'gaussian_kde'}")
            
            # Categorical features - store probabilities
            for feature in self.categorical_features:
                probabilities = subset[feature].value_counts(normalize=True).to_dict()
                self.target_distributions[target_value][feature] = {
                    'probabilities': probabilities,
                    'original_data': subset[feature].values
                }
                
                prob_str = ", ".join([f"{k}:{v:.2f}" for k, v in probabilities.items()])
                print(f"     {feature:25} Probabilities: {prob_str}")
    
    def analyze_correlations(self):
        """Analyze feature correlations to preserve in synthetic data."""
        if self.original_df is None:
            return None
            
        print(f"\n🔗 ANALYZING FEATURE CORRELATIONS")
        print("="*35)
        
        # Calculate correlation matrix for numerical features
        numerical_data = self.original_df[self.numerical_features]
        self.correlation_matrix = numerical_data.corr()
        
        # Identify strong correlations (|r| > 0.3)
        strong_correlations = []
        for i, col1 in enumerate(self.correlation_matrix.columns):
            for j, col2 in enumerate(self.correlation_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_value = self.correlation_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.3:
                        strong_correlations.append((col1, col2, corr_value))
        
        print(f"   Found {len(strong_correlations)} strong correlations to preserve:")
        for feat1, feat2, corr in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
            print(f"     {feat1} ↔ {feat2}: r = {corr:.3f}")
        
        return strong_correlations
    
    def generate_stratified_synthetic_data(self, n_synthetic=701):
        """
        Generate synthetic data maintaining class distribution.
        
        Args:
            n_synthetic: Number of synthetic records to generate (default 701)
            
        Returns:
            DataFrame with synthetic records
        """
        if self.original_df is None:
            return None
            
        print(f"\n🏭 GENERATING {n_synthetic} SYNTHETIC RECORDS")
        print("="*45)
        
        # Calculate target distribution in original data
        original_target_dist = self.original_df[self.target].value_counts(normalize=True)
        print(f"   Original target distribution: {original_target_dist.to_dict()}")
        
        # Calculate number of synthetic records per class
        n_survived = int(n_synthetic * original_target_dist[0])
        n_died = n_synthetic - n_survived
        
        print(f"   Generating {n_survived} survived + {n_died} died = {n_synthetic} total")
        
        synthetic_records = []
        
        # Generate for each target class
        for target_value, n_samples in [(0, n_survived), (1, n_died)]:
            target_name = "Survived" if target_value == 0 else "Died"
            print(f"\n   📈 Generating {n_samples} {target_name} records...")
            
            for i in range(n_samples):
                record = {'DEATH_EVENT': target_value}
                
                # Generate numerical features
                for feature in self.numerical_features:
                    dist_info = self.target_distributions[target_value][feature]
                    
                    if dist_info['distribution'] and dist_info['params']:
                        # Use fitted distribution
                        value = dist_info['distribution'].rvs(*dist_info['params'])
                    else:
                        # Fallback to gaussian KDE
                        kde = gaussian_kde(dist_info['original_data'])
                        value = kde.resample(1)[0][0]
                    
                    # Apply bounds based on original data
                    original_min = self.original_df[feature].min()
                    original_max = self.original_df[feature].max()
                    value = np.clip(value, original_min, original_max)
                    
                    record[feature] = value
                
                # Generate categorical features
                for feature in self.categorical_features:
                    probabilities = self.target_distributions[target_value][feature]['probabilities']
                    
                    # Sample based on probabilities
                    values = list(probabilities.keys())
                    probs = list(probabilities.values())
                    value = np.random.choice(values, p=probs)
                    
                    record[feature] = value
                
                synthetic_records.append(record)
        
        # Create synthetic DataFrame
        self.synthetic_df = pd.DataFrame(synthetic_records)
        
        # Reorder columns to match original
        self.synthetic_df = self.synthetic_df[self.original_df.columns]
        
        print(f"   ✅ Generated {len(self.synthetic_df)} synthetic records")
        
        return self.synthetic_df
    
    def improve_correlation_preservation(self):
        """
        Improve correlation preservation using multivariate techniques.
        """
        if self.synthetic_df is None:
            return None
            
        print("\n🔧 IMPROVING CORRELATION PRESERVATION")
        print("="*40)
        
        # For each target class, adjust correlations
        for target_value in [0, 1]:
            target_name = "Survived" if target_value == 0 else "Died"
            
            # Original data subset
            original_subset = self.original_df[self.original_df[self.target] == target_value]
            original_numerical = original_subset[self.numerical_features]
            
            # Synthetic data subset  
            synthetic_subset = self.synthetic_df[self.synthetic_df[self.target] == target_value]
            synthetic_numerical = synthetic_subset[self.numerical_features].copy()
            
            # Calculate correlation matrices
            original_corr = original_numerical.corr()
            synthetic_corr = synthetic_numerical.corr()
            
            # Use Cholesky decomposition to adjust correlations
            try:
                # Standardize synthetic data
                scaler = StandardScaler()
                synthetic_scaled = scaler.fit_transform(synthetic_numerical)
                
                # Generate correlated data using Cholesky decomposition
                target_corr = original_corr.values
                L = np.linalg.cholesky(target_corr)
                
                # Generate uncorrelated data and transform
                uncorr_data = np.random.randn(len(synthetic_scaled), len(self.numerical_features))
                correlated_data = uncorr_data @ L.T
                
                # Scale back to original distribution characteristics
                for i, feature in enumerate(self.numerical_features):
                    original_mean = original_numerical[feature].mean()
                    original_std = original_numerical[feature].std()
                    
                    correlated_data[:, i] = correlated_data[:, i] * original_std + original_mean
                    
                    # Apply bounds
                    original_min = self.original_df[feature].min()
                    original_max = self.original_df[feature].max()
                    correlated_data[:, i] = np.clip(correlated_data[:, i], original_min, original_max)
                
                # Update synthetic data
                synthetic_indices = self.synthetic_df[self.synthetic_df[self.target] == target_value].index
                for i, feature in enumerate(self.numerical_features):
                    self.synthetic_df.loc[synthetic_indices, feature] = correlated_data[:, i]
                
                print(f"   ✅ Improved correlations for {target_name} group")
                
            except np.linalg.LinAlgError:
                print(f"   ⚠️ Could not improve correlations for {target_name} group (matrix not positive definite)")
        
        return self.synthetic_df
    
    def validate_synthetic_data_quality(self):
        """
        Validate the quality of synthetic data against original data.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.synthetic_df is None or self.original_df is None:
            return None
            
        print("\n🔍 VALIDATING SYNTHETIC DATA QUALITY")
        print("="*40)
        
        validation_results = {}
        
        # 1. Statistical similarity for numerical features
        print("\n   📊 STATISTICAL SIMILARITY:")
        for feature in self.numerical_features:
            original_mean = self.original_df[feature].mean()
            original_std = self.original_df[feature].std()
            synthetic_mean = self.synthetic_df[feature].mean()
            synthetic_std = self.synthetic_df[feature].std()
            
            mean_diff = abs(synthetic_mean - original_mean) / original_mean * 100
            std_diff = abs(synthetic_std - original_std) / original_std * 100
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(
                self.original_df[feature], 
                self.synthetic_df[feature]
            )
            
            validation_results[feature] = {
                'mean_diff_pct': mean_diff,
                'std_diff_pct': std_diff,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'distribution_similar': ks_pvalue > 0.05
            }
            
            similar_mark = "✅" if ks_pvalue > 0.05 else "⚠️"
            print(f"     {similar_mark} {feature:25} Mean: {mean_diff:.1f}%, Std: {std_diff:.1f}%, KS p-value: {ks_pvalue:.3f}")
        
        # 2. Categorical feature distribution similarity
        print(f"\n   🏷️ CATEGORICAL SIMILARITY:")
        for feature in self.categorical_features:
            original_dist = self.original_df[feature].value_counts(normalize=True).sort_index()
            synthetic_dist = self.synthetic_df[feature].value_counts(normalize=True).sort_index()
            
            # Chi-square test for distribution similarity
            from scipy.stats import chisquare
            chi2_stat, chi2_pvalue = chisquare(synthetic_dist.values, original_dist.values)
            
            validation_results[f"{feature}_categorical"] = {
                'chi2_statistic': chi2_stat,
                'chi2_pvalue': chi2_pvalue,
                'distribution_similar': chi2_pvalue > 0.05
            }
            
            similar_mark = "✅" if chi2_pvalue > 0.05 else "⚠️"
            print(f"     {similar_mark} {feature:25} Chi-square p-value: {chi2_pvalue:.3f}")
        
        # 3. Target distribution preservation
        original_target_dist = self.original_df[self.target].value_counts(normalize=True)
        synthetic_target_dist = self.synthetic_df[self.target].value_counts(normalize=True)
        
        target_diff = abs(synthetic_target_dist - original_target_dist).max()
        
        print(f"\n   🎯 TARGET DISTRIBUTION:")
        print(f"     Original: {original_target_dist.to_dict()}")
        print(f"     Synthetic: {synthetic_target_dist.to_dict()}")
        print(f"     Max difference: {target_diff:.3f}")
        
        validation_results['target_distribution'] = {
            'max_difference': target_diff,
            'well_preserved': target_diff < 0.05
        }
        
        # Overall quality score
        similar_distributions = sum([
            1 for feature in self.numerical_features 
            if validation_results[feature]['distribution_similar']
        ])
        
        quality_score = similar_distributions / len(self.numerical_features) * 100
        
        print(f"\n   🏆 OVERALL QUALITY SCORE: {quality_score:.1f}%")
        print(f"       ({similar_distributions}/{len(self.numerical_features)} features have similar distributions)")
        
        validation_results['overall_quality_score'] = quality_score
        
        return validation_results
    
    def create_combined_dataset(self):
        """
        Combine original and synthetic data into final dataset.
        
        Returns:
            Combined DataFrame with 1000 records (299 original + 701 synthetic)
        """
        if self.original_df is None or self.synthetic_df is None:
            return None
            
        print("\n🔗 CREATING COMBINED DATASET")
        print("="*30)
        
        # Add source indicator
        original_marked = self.original_df.copy()
        original_marked['data_source'] = 'original'
        
        synthetic_marked = self.synthetic_df.copy()
        synthetic_marked['data_source'] = 'synthetic'
        
        # Combine datasets
        self.combined_df = pd.concat([original_marked, synthetic_marked], ignore_index=True)
        
        # Shuffle the combined dataset
        self.combined_df = self.combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   ✅ Combined dataset created: {self.combined_df.shape}")
        print(f"     Original records: {len(original_marked)}")
        print(f"     Synthetic records: {len(synthetic_marked)}")
        print(f"     Total records: {len(self.combined_df)}")
        
        # Final target distribution
        final_target_dist = self.combined_df[self.target].value_counts()
        print(f"     Final target distribution: {final_target_dist.to_dict()}")
        
        return self.combined_df
    
    def save_synthetic_datasets(self):
        """Save synthetic and combined datasets."""
        print("\n💾 SAVING DATASETS")
        print("="*20)
        
        # Create output directory
        import os
        os.makedirs("../data/synthetic", exist_ok=True)
        os.makedirs("../data/processed", exist_ok=True)
        
        # Save synthetic data only
        if self.synthetic_df is not None:
            self.synthetic_df.to_csv("../data/synthetic/synthetic_heart_failure_701.csv", index=False)
            print("   ✅ Synthetic data saved: ../data/synthetic/synthetic_heart_failure_701.csv")
        
        # Save combined dataset  
        if self.combined_df is not None:
            # Without source indicator for modeling
            modeling_dataset = self.combined_df.drop('data_source', axis=1)
            modeling_dataset.to_csv("../data/processed/heart_failure_combined_1000.csv", index=False)
            print("   ✅ Combined data saved: ../data/processed/heart_failure_combined_1000.csv")
            
            # With source indicator for analysis
            self.combined_df.to_csv("../data/processed/heart_failure_with_source.csv", index=False)
            print("   ✅ Data with source labels saved: ../data/processed/heart_failure_with_source.csv")


def main():
    """
    Main function to execute Phase 3 synthetic data generation workflow.
    
    Team Usage:
    - Jose: Lead synthetic data generation and validation
    - Julian: Support with statistical validation and quality assessment
    """
    print("="*60)
    print("PHASE 3: SYNTHETIC DATA GENERATION")
    print("Team Lead: Jose (Synthetic Data Generator)")
    print("Goal: Expand dataset from 299 to 1000 records")
    print("="*60)
    
    # Initialize synthetic data generator
    generator = SyntheticDataGenerator()
    
    if generator.original_df is not None:
        # Execute synthetic data generation pipeline
        print("\n🏭 Starting synthetic data generation pipeline...")
        
        # 1. Generate stratified synthetic data
        synthetic_data = generator.generate_stratified_synthetic_data(n_synthetic=701)
        
        # 2. Improve correlation preservation
        improved_data = generator.improve_correlation_preservation()
        
        # 3. Validate synthetic data quality
        validation_results = generator.validate_synthetic_data_quality()
        
        # 4. Create combined dataset
        combined_data = generator.create_combined_dataset()
        
        # 5. Save datasets
        generator.save_synthetic_datasets()
        
        print("\n✅ Phase 3 synthetic data generation completed!")
        print("📊 Dataset expanded from 299 to 1000 records")
        print("🎯 Ready for Phase 4: Decision Tree model training")
        
    else:
        print("❌ Cannot proceed without original data. Check data loading.")


if __name__ == "__main__":
    main()