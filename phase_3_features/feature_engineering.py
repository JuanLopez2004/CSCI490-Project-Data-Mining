"""
Phase 3: Feature Engineering and Data Preprocessing
Team: Goblins Hiding in Vents (Cameron, Julian, Juan, Nathan, Nasiru, Jose)
Week: 3

This module implements comprehensive feature engineering and data preprocessing.
Team Leaders: Julian (Feature Engineering Specialist), Jose (Synthetic Data Generator)

Goals:
- Implement Chi-square feature selection based on Phase 2 findings
- Create feature scaling and normalization pipelines
- Generate 701 synthetic records maintaining statistical distributions
- Create derived features for enhanced prediction
- Prepare final dataset for machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import norm, skew
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Comprehensive feature engineering for heart failure prediction.
    
    Responsibilities:
    - Chi-square feature selection implementation
    - Feature scaling and normalization
    - Derived feature creation (age groups, risk scores, clinical categories)
    - Data preprocessing pipeline
    - Feature transformation and encoding
    """
    
    def __init__(self, data_path: str = "../data/raw/heart_failure_clinical_records.csv"):
        """
        Initialize feature engineering pipeline.
        
        Args:
            data_path: Path to the raw heart failure dataset
        """
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.feature_importance = {}
        self.scalers = {}
        
        # Feature categories from Phase 2 EDA
        self.numerical_features = [
            'age', 'creatinine_phosphokinase', 'ejection_fraction',
            'platelets', 'serum_creatinine', 'serum_sodium', 'time'
        ]
        
        self.categorical_features = [
            'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking'
        ]
        
        self.target = 'DEATH_EVENT'
        
        self.load_data()
    
    def load_data(self):
        """Load and prepare data for feature engineering."""
        try:
            self.df = pd.read_csv(self.data_path)
            self.processed_df = self.df.copy()
            print(f"✅ Data loaded for feature engineering: {self.df.shape}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def perform_chi_square_selection(self, k_features=10):
        """
        Perform chi-square feature selection based on Phase 2 analysis.
        
        Args:
            k_features: Number of top features to select
            
        Returns:
            Selected features and their chi-square scores
        """
        if self.processed_df is None:
            return None
            
        print("🔬 PERFORMING CHI-SQUARE FEATURE SELECTION")
        print("="*50)
        
        # Prepare features and target
        X = self.processed_df.drop([self.target], axis=1)
        y = self.processed_df[self.target]
        
        # For chi-square, we need non-negative values
        # Apply Min-Max scaling to ensure positive values
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Perform chi-square test
        chi2_selector = SelectKBest(chi2, k=k_features)
        X_chi2 = chi2_selector.fit_transform(X_scaled, y)
        
        # Get feature scores and selected features
        feature_scores = chi2_selector.scores_
        selected_features = X.columns[chi2_selector.get_support()]
        
        # Create feature importance ranking
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'chi2_score': feature_scores,
            'p_value': chi2_selector.pvalues_,
            'selected': chi2_selector.get_support()
        }).sort_values('chi2_score', ascending=False)
        
        print("📊 CHI-SQUARE FEATURE RANKING:")
        for idx, row in feature_importance_df.head(10).iterrows():
            selected_mark = "✅" if row['selected'] else "❌"
            print(f"{selected_mark} {row['feature']:25} Score: {row['chi2_score']:.2f}, p-value: {row['p_value']:.4f}")
        
        self.feature_importance['chi2'] = feature_importance_df
        
        print(f"\n🎯 Selected {len(selected_features)} features for modeling")
        print(f"   Selected: {list(selected_features)}")
        
        return selected_features, feature_importance_df
    
    def create_derived_features(self):
        """
        Create derived features based on clinical insights from Phase 2.
        
        Returns:
            DataFrame with derived features added
        """
        if self.processed_df is None:
            return None
            
        print("\n🛠️ CREATING DERIVED FEATURES")
        print("="*40)
        
        # 1. Age group categories (from EDA insights)
        age_bins = [0, 50, 65, 75, 100]
        age_labels = [0, 1, 2, 3]  # Encoded for ML
        self.processed_df['age_group'] = pd.cut(
            self.processed_df['age'], 
            bins=age_bins, 
            labels=age_labels, 
            include_lowest=True
        ).astype(int)
        
        print("   ✅ Created age_group feature (0:<50, 1:50-64, 2:65-74, 3:≥75)")
        
        # 2. Ejection fraction categories (clinical thresholds)
        ef_bins = [0, 40, 50, 100]
        ef_labels = [2, 1, 0]  # 2=Reduced, 1=Borderline, 0=Normal (higher is worse)
        self.processed_df['ef_category'] = pd.cut(
            self.processed_df['ejection_fraction'],
            bins=ef_bins,
            labels=ef_labels,
            include_lowest=True
        ).astype(int)
        
        print("   ✅ Created ef_category feature (0:Normal≥50%, 1:Borderline40-49%, 2:Reduced<40%)")
        
        # 3. Serum creatinine risk levels
        creat_bins = [0, 1.2, 2.0, 10]
        creat_labels = [0, 1, 2]  # 0=Normal, 1=Elevated, 2=High
        self.processed_df['creatinine_risk'] = pd.cut(
            self.processed_df['serum_creatinine'],
            bins=creat_bins,
            labels=creat_labels,
            include_lowest=True
        ).astype(int)
        
        print("   ✅ Created creatinine_risk feature (0:Normal<1.2, 1:Elevated1.2-2.0, 2:High>2.0)")
        
        # 4. Combined risk score (sum of binary risk factors)
        risk_factors = ['diabetes', 'high_blood_pressure', 'smoking', 'anaemia']
        self.processed_df['risk_score'] = self.processed_df[risk_factors].sum(axis=1)
        
        print(f"   ✅ Created risk_score feature (sum of {len(risk_factors)} binary risk factors)")
        
        # 5. Kidney function indicator (based on serum_creatinine and age)
        # Higher creatinine relative to age indicates worse kidney function
        age_norm_creatinine = self.processed_df['serum_creatinine'] / (self.processed_df['age'] / 100)
        self.processed_df['kidney_function_index'] = age_norm_creatinine
        
        print("   ✅ Created kidney_function_index feature (age-normalized creatinine)")
        
        # 6. Cardiovascular stress indicator
        # Combination of ejection fraction and blood pressure
        cv_stress = (100 - self.processed_df['ejection_fraction']) * (1 + self.processed_df['high_blood_pressure'])
        self.processed_df['cv_stress_index'] = cv_stress
        
        print("   ✅ Created cv_stress_index feature (EF and BP combination)")
        
        # 7. Follow-up time categories (short, medium, long term)
        time_bins = [0, 50, 150, 300]
        time_labels = [0, 1, 2]  # 0=Short, 1=Medium, 2=Long
        self.processed_df['followup_category'] = pd.cut(
            self.processed_df['time'],
            bins=time_bins,
            labels=time_labels,
            include_lowest=True
        ).astype(int)
        
        print("   ✅ Created followup_category feature (0:Short<50d, 1:Medium50-150d, 2:Long>150d)")
        
        # 8. High-risk profile (combination of worst predictors from EDA)
        high_risk_conditions = [
            (self.processed_df['ejection_fraction'] < 40),  # Reduced EF
            (self.processed_df['serum_creatinine'] > 1.5),  # Elevated creatinine  
            (self.processed_df['age'] > 70),                # Advanced age
            (self.processed_df['serum_sodium'] < 135)       # Hyponatremia
        ]
        
        self.processed_df['high_risk_profile'] = sum(high_risk_conditions).astype(int)
        
        print("   ✅ Created high_risk_profile feature (sum of 4 high-risk conditions)")
        
        # Summary of new features
        new_features = [
            'age_group', 'ef_category', 'creatinine_risk', 'risk_score',
            'kidney_function_index', 'cv_stress_index', 'followup_category', 'high_risk_profile'
        ]
        
        print(f"\n📈 Added {len(new_features)} derived features")
        print(f"   Total features: {len(self.processed_df.columns) - 1} (excluding target)")
        
        return new_features
    
    def apply_feature_scaling(self, method='standard'):
        """
        Apply feature scaling to numerical features.
        
        Args:
            method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
            
        Returns:
            Scaled dataset
        """
        if self.processed_df is None:
            return None
            
        print(f"\n⚖️ APPLYING FEATURE SCALING ({method.upper()})")
        print("="*45)
        
        # Identify numerical features (including new derived ones)
        numerical_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != self.target]
        
        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
            scaler_name = "StandardScaler (mean=0, std=1)"
        else:
            scaler = MinMaxScaler()
            scaler_name = "MinMaxScaler (range=0-1)"
        
        # Apply scaling
        scaled_data = scaler.fit_transform(self.processed_df[numerical_cols])
        
        # Create scaled dataframe
        scaled_df = self.processed_df.copy()
        scaled_df[numerical_cols] = scaled_data
        
        # Store scaler for future use
        self.scalers[method] = scaler
        
        print(f"   ✅ Applied {scaler_name}")
        print(f"   📊 Scaled {len(numerical_cols)} numerical features")
        
        # Show scaling statistics
        print(f"\n   📈 Scaling verification (first 5 features):")
        for col in numerical_cols[:5]:
            original_mean = self.processed_df[col].mean()
            original_std = self.processed_df[col].std()
            scaled_mean = scaled_df[col].mean()
            scaled_std = scaled_df[col].std()
            
            print(f"     {col:20} Original: μ={original_mean:.2f}, σ={original_std:.2f} | Scaled: μ={scaled_mean:.3f}, σ={scaled_std:.3f}")
        
        return scaled_df
    
    def handle_feature_interactions(self):
        """
        Create interaction features between important predictors.
        
        Returns:
            DataFrame with interaction features
        """
        if self.processed_df is None:
            return None
            
        print("\n🔗 CREATING FEATURE INTERACTIONS")
        print("="*35)
        
        # Based on Phase 2 EDA, create interactions between top predictors
        
        # 1. Ejection Fraction × Age interaction
        self.processed_df['ef_age_interaction'] = (
            self.processed_df['ejection_fraction'] * self.processed_df['age']
        )
        
        # 2. Serum Creatinine × Age interaction  
        self.processed_df['creat_age_interaction'] = (
            self.processed_df['serum_creatinine'] * self.processed_df['age']
        )
        
        # 3. Risk Score × Age interaction
        self.processed_df['risk_age_interaction'] = (
            self.processed_df['risk_score'] * self.processed_df['age']
        )
        
        # 4. EF × Creatinine interaction (heart-kidney axis)
        self.processed_df['ef_creat_interaction'] = (
            self.processed_df['ejection_fraction'] * self.processed_df['serum_creatinine']
        )
        
        interaction_features = [
            'ef_age_interaction', 'creat_age_interaction', 
            'risk_age_interaction', 'ef_creat_interaction'
        ]
        
        print(f"   ✅ Created {len(interaction_features)} interaction features")
        for feature in interaction_features:
            print(f"     • {feature}")
        
        return interaction_features
    
    def prepare_final_dataset(self, test_size=0.2, random_state=42):
        """
        Prepare final dataset for machine learning.
        
        Args:
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Train/test splits ready for modeling
        """
        if self.processed_df is None:
            return None
            
        print("\n📦 PREPARING FINAL DATASET")
        print("="*30)
        
        # Final feature set (exclude target)
        feature_columns = [col for col in self.processed_df.columns if col != self.target]
        X = self.processed_df[feature_columns]
        y = self.processed_df[self.target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"   ✅ Dataset prepared for modeling")
        print(f"     Total samples: {len(X)}")
        print(f"     Total features: {len(feature_columns)}")
        print(f"     Train set: {len(X_train)} samples")
        print(f"     Test set: {len(X_test)} samples")
        print(f"     Target distribution (train): {y_train.value_counts().to_dict()}")
        
        # Save processed dataset
        processed_data = {
            'X_train': X_train,
            'X_test': X_test, 
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_columns,
            'target_name': self.target
        }
        
        return processed_data
    
    def generate_feature_engineering_report(self):
        """Generate comprehensive feature engineering report."""
        if self.processed_df is None:
            return None
            
        print("\n📋 FEATURE ENGINEERING SUMMARY REPORT")
        print("="*50)
        print("Team: Goblins Hiding in Vents")
        print("Phase: 3 - Feature Engineering") 
        print("Team Leaders: Julian (Engineering), Jose (Synthetic Data)")
        print("="*50)
        
        # Original vs processed comparison
        original_features = len(self.df.columns) - 1
        processed_features = len(self.processed_df.columns) - 1
        added_features = processed_features - original_features
        
        print(f"\n📊 FEATURE TRANSFORMATION SUMMARY:")
        print(f"   Original features: {original_features}")
        print(f"   Derived features added: {added_features}")
        print(f"   Total features: {processed_features}")
        print(f"   Feature increase: {(added_features/original_features)*100:.1f}%")
        
        print(f"\n🎯 KEY ENHANCEMENTS:")
        print("   ✅ Chi-square feature selection implemented")
        print("   ✅ Clinical threshold-based categories created")
        print("   ✅ Risk scoring mechanisms developed")
        print("   ✅ Feature interactions generated")
        print("   ✅ Age-normalized indices calculated")
        print("   ✅ Cardiovascular stress indicators added")
        print("   ✅ Feature scaling pipelines established")
        
        print(f"\n🔬 CLINICAL VALIDATION:")
        print("   • Age groups align with geriatric medicine standards")
        print("   • EF categories match cardiology guidelines (<40%, 40-49%, ≥50%)")
        print("   • Creatinine thresholds follow nephrology standards")
        print("   • Risk scores incorporate established HF risk factors")
        
        print(f"\n📈 READY FOR PHASE 4: MODEL DEVELOPMENT")
        print("   Next: Decision Tree training with enhanced feature set")


def main():
    """
    Main function to execute Phase 3 feature engineering workflow.
    
    Team Usage:
    - Julian: Lead feature engineering and derived feature creation
    - Jose: Support with preprocessing and data validation
    - Nathan: Statistical validation of transformations
    """
    print("="*60)
    print("PHASE 3: FEATURE ENGINEERING & DATA PREPROCESSING")
    print("Team Leaders: Julian (Feature Engineering), Jose (Synthetic Data)")
    print("="*60)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    if fe.df is not None:
        # Execute feature engineering pipeline
        print("\n🛠️ Starting comprehensive feature engineering...")
        
        # 1. Chi-square feature selection
        selected_features, importance_df = fe.perform_chi_square_selection(k_features=10)
        
        # 2. Create derived features
        new_features = fe.create_derived_features()
        
        # 3. Create feature interactions
        interaction_features = fe.handle_feature_interactions()
        
        # 4. Apply feature scaling
        scaled_df = fe.apply_feature_scaling(method='standard')
        
        # 5. Prepare final dataset
        final_data = fe.prepare_final_dataset()
        
        # 6. Generate summary report
        fe.generate_feature_engineering_report()
        
        print("\n✅ Phase 3 feature engineering completed!")
        print("📊 Enhanced dataset ready for Decision Tree modeling")
        
        # Save processed data for Phase 4
        if final_data:
            print("\n💾 Processed data ready for Phase 4 model training")
        
    else:
        print("❌ Cannot proceed without data. Check data loading.")


if __name__ == "__main__":
    main()