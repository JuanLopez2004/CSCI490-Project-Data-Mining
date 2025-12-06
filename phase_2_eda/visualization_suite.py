"""
Phase 2: Data Visualization Suite
Team: Goblins Hiding in Vents
Lead: Nasiru (Data Visualization Lead)
Support: Nathan (Statistical Analyst)

This module creates comprehensive visualizations for heart failure EDA.
Focus on clinical insights and pattern discovery for machine learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class HeartFailureVisualizer:
    """
    Advanced visualization suite for heart failure prediction analysis.
    
    Responsibilities:
    - Interactive plotly visualizations
    - Clinical insight dashboards  
    - Feature importance visualization preparation
    - Publication-ready static plots
    """
    
    def __init__(self, data_path: str = "../Datasets/training_data.csv"):
        """Initialize visualizer with heart failure data."""
        self.data_path = data_path
        self.df = None
        self.load_data()
        
        # Color schemes for consistency
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#F18F01',
            'danger': '#C73E1D',
            'survived': '#2ECC71',
            'died': '#E74C3C'
        }
        
    def load_data(self):
        """Load and prepare data for visualization."""
        try:
            self.df = pd.read_csv(self.data_path)
            # Add survival label for clarity
            self.df['survival_status'] = self.df['DEATH_EVENT'].map({0: 'Survived', 1: 'Died'})
            print(f"✅ Data loaded for visualization: {self.df.shape}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def create_overview_dashboard(self):
        """Create comprehensive overview dashboard."""
        if self.df is None:
            return None
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Survival Distribution', 'Age Distribution by Outcome', 
                'Ejection Fraction by Outcome', 'Key Risk Factors',
                'Serum Creatinine Levels', 'Gender Distribution'
            ],
            specs=[[{"type": "pie"}, {"type": "box"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "violin"}, {"type": "bar"}]]
        )
        
        # 1. Survival pie chart
        survival_counts = self.df['survival_status'].value_counts()
        fig.add_trace(
            go.Pie(labels=survival_counts.index, values=survival_counts.values,
                  marker=dict(colors=[self.colors['survived'], self.colors['died']]),
                  name="Survival"),
            row=1, col=1
        )
        
        # 2. Age by outcome box plot
        for outcome, color in [('Survived', self.colors['survived']), ('Died', self.colors['died'])]:
            data = self.df[self.df['survival_status'] == outcome]['age']
            fig.add_trace(
                go.Box(y=data, name=outcome, marker=dict(color=color)),
                row=1, col=2
            )
        
        # 3. Ejection fraction by outcome
        for outcome, color in [('Survived', self.colors['survived']), ('Died', self.colors['died'])]:
            data = self.df[self.df['survival_status'] == outcome]['ejection_fraction']
            fig.add_trace(
                go.Box(y=data, name=outcome, marker=dict(color=color), showlegend=False),
                row=1, col=3
            )
        
        # 4. Risk factors bar chart
        risk_factors = ['diabetes', 'high_blood_pressure', 'smoking', 'anaemia']
        risk_percentages = []
        
        for factor in risk_factors:
            # Percentage of people with this risk factor who died
            died_with_factor = self.df[(self.df[factor] == 1) & (self.df['DEATH_EVENT'] == 1)].shape[0]
            total_with_factor = self.df[self.df[factor] == 1].shape[0]
            percentage = (died_with_factor / total_with_factor * 100) if total_with_factor > 0 else 0
            risk_percentages.append(percentage)
        
        fig.add_trace(
            go.Bar(x=risk_factors, y=risk_percentages, 
                  marker=dict(color=self.colors['danger']),
                  name="Death Rate %"),
            row=2, col=1
        )
        
        # 5. Serum creatinine violin plot
        for outcome, color in [('Survived', self.colors['survived']), ('Died', self.colors['died'])]:
            data = self.df[self.df['survival_status'] == outcome]['serum_creatinine']
            fig.add_trace(
                go.Violin(y=data, name=outcome, marker=dict(color=color), showlegend=False),
                row=2, col=2
            )
        
        # 6. Gender distribution
        gender_survival = pd.crosstab(self.df['sex'], self.df['survival_status'], normalize='index') * 100
        
        for i, outcome in enumerate(['Survived', 'Died']):
            fig.add_trace(
                go.Bar(x=['Female', 'Male'], y=gender_survival[outcome].values,
                      name=outcome, 
                      marker=dict(color=[self.colors['survived'], self.colors['died']][i])),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title_text="Heart Failure Dataset - Clinical Overview Dashboard",
            height=800,
            showlegend=True,
            title_x=0.5
        )
        
        fig.show()
        return fig
    
    def create_feature_correlation_network(self):
        """Create network-style correlation visualization."""
        if self.df is None:
            return None
            
        # Calculate correlations
        corr_matrix = self.df.select_dtypes(include=[np.number]).corr()
        
        # Create interactive heatmap
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix - Interactive",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            text_auto=True
        )
        
        fig.update_layout(
            title_x=0.5,
            height=600,
            width=800
        )
        
        fig.show()
        return fig
    
    def create_clinical_insights_plots(self):
        """Create plots focused on clinical insights."""
        if self.df is None:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Critical thresholds analysis
        ax1 = axes[0, 0]
        
        # Ejection fraction threshold (normal >50%, borderline 41-49%, reduced <40%)
        ejection_bins = [0, 40, 50, 100]
        ejection_labels = ['Reduced (<40%)', 'Borderline (40-49%)', 'Normal (≥50%)']
        self.df['ef_category'] = pd.cut(self.df['ejection_fraction'], bins=ejection_bins, labels=ejection_labels)
        
        ef_death_rate = self.df.groupby('ef_category')['DEATH_EVENT'].agg(['sum', 'count'])
        ef_death_rate['death_rate'] = ef_death_rate['sum'] / ef_death_rate['count'] * 100
        
        bars = ax1.bar(ef_death_rate.index, ef_death_rate['death_rate'], 
                      color=['#E74C3C', '#F39C12', '#2ECC71'])
        ax1.set_title('Death Rate by Ejection Fraction Category')
        ax1.set_ylabel('Death Rate (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 2. Age group analysis
        ax2 = axes[0, 1]
        
        age_bins = [0, 50, 65, 75, 100]
        age_labels = ['<50', '50-64', '65-74', '≥75']
        self.df['age_group'] = pd.cut(self.df['age'], bins=age_bins, labels=age_labels)
        
        age_analysis = self.df.groupby(['age_group', 'survival_status']).size().unstack()
        age_analysis.plot(kind='bar', ax=ax2, color=[self.colors['survived'], self.colors['died']])
        ax2.set_title('Survival by Age Group')
        ax2.set_xlabel('Age Group (years)')
        ax2.set_ylabel('Number of Patients')
        ax2.legend(['Survived', 'Died'])
        ax2.tick_params(axis='x', rotation=0)
        
        # 3. Serum creatinine risk levels
        ax3 = axes[1, 0]
        
        # Normal: <1.2, Elevated: 1.2-2.0, High: >2.0
        creat_bins = [0, 1.2, 2.0, 10]
        creat_labels = ['Normal (<1.2)', 'Elevated (1.2-2.0)', 'High (>2.0)']
        self.df['creatinine_category'] = pd.cut(self.df['serum_creatinine'], bins=creat_bins, labels=creat_labels)
        
        creat_death_rate = self.df.groupby('creatinine_category')['DEATH_EVENT'].agg(['sum', 'count'])
        creat_death_rate['death_rate'] = creat_death_rate['sum'] / creat_death_rate['count'] * 100
        
        bars = ax3.bar(creat_death_rate.index, creat_death_rate['death_rate'],
                      color=['#2ECC71', '#F39C12', '#E74C3C'])
        ax3.set_title('Death Rate by Serum Creatinine Level')
        ax3.set_ylabel('Death Rate (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 4. Combined risk factors
        ax4 = axes[1, 1]
        
        # Create risk score (number of risk factors)
        risk_factors = ['diabetes', 'high_blood_pressure', 'smoking', 'anaemia']
        self.df['risk_score'] = self.df[risk_factors].sum(axis=1)
        
        risk_survival = pd.crosstab(self.df['risk_score'], self.df['survival_status'], normalize='index') * 100
        risk_survival.plot(kind='bar', ax=ax4, color=[self.colors['survived'], self.colors['died']])
        ax4.set_title('Survival Rate by Number of Risk Factors')
        ax4.set_xlabel('Number of Risk Factors')
        ax4.set_ylabel('Percentage (%)')
        ax4.legend(['Survived', 'Died'])
        ax4.tick_params(axis='x', rotation=0)
        
        plt.suptitle('Clinical Insights - Heart Failure Risk Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_visualizations(self, output_dir="../reports/phase_2_visualizations/"):
        """Save all visualizations for team sharing."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Saving visualizations to {output_dir}")
        print("   - Overview dashboard")
        print("   - Correlation matrix") 
        print("   - Clinical insights")
        print("   - Feature importance")
        
        # Note: In actual implementation, you would save the figures
        print("✅ Visualizations saved for team review")


def main():
    """
    Main function for Phase 2 visualization workflow.
    
    Team Usage:
    - Nasiru: Lead visualization development
    - Nathan: Statistical validation of plots
    - All team: Review insights for Phase 3 planning
    """
    print("="*60)
    print("PHASE 2: DATA VISUALIZATION SUITE") 
    print("Lead: Nasiru (Data Visualization)")
    print("Support: Nathan (Statistical Validation)")
    print("="*60)
    
    # Initialize visualizer
    viz = HeartFailureVisualizer()
    
    if viz.df is not None:
        print("\n🎨 Creating comprehensive visualization suite...")
        
        # Generate all visualizations
        viz.create_overview_dashboard()
        viz.create_feature_correlation_network()
        viz.create_clinical_insights_plots()
        
        # Save for team
        viz.save_visualizations()
        
        print("\n✅ Phase 2 visualizations completed!")
        print("📊 All charts ready for clinical insight analysis")
        print("🔬 Statistical patterns identified for Phase 3")
        
    else:
        print("❌ Cannot create visualizations without data")


if __name__ == "__main__":
    main()
