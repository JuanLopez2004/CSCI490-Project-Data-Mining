"""
Phase 1: Project Setup and Team Organization
Team: Goblins Hiding in Vents

This file contains team member roles, responsibilities, and project setup guidelines.
"""

# TEAM INFORMATION
TEAM_NAME = "Goblins Hiding in Vents"
TEAM_MEMBERS = {
    "Cameron": {
        "role": "Data Collection Lead",
        "responsibilities": [
            "UCI dataset acquisition and validation",
            "Data quality assessment",
            "Initial data exploration setup"
        ],
        "phase_focus": "Phase 1 & 2"
    },
    "Julian": {
        "role": "Feature Engineering Specialist", 
        "responsibilities": [
            "Feature selection and creation",
            "Data preprocessing pipeline",
            "Statistical analysis of features"
        ],
        "phase_focus": "Phase 3 & 4"
    },
    "Juan": {
        "role": "Machine Learning Engineer",
        "responsibilities": [
            "Decision tree model development",
            "Hyperparameter tuning",
            "Cross-validation implementation"
        ],
        "phase_focus": "Phase 4 & 5"
    },
    "Nathan": {
        "role": "Statistical Analyst",
        "responsibilities": [
            "Chi-square analysis",
            "Bootstrap sampling",
            "ROC curve generation and analysis"
        ],
        "phase_focus": "Phase 2 & 5"
    },
    "Nasiru": {
        "role": "Data Visualization Lead",
        "responsibilities": [
            "EDA visualizations",
            "Model performance charts",
            "Decision tree visualization"
        ],
        "phase_focus": "Phase 2 & 6"
    },
    "Jose": {
        "role": "Synthetic Data Generator",
        "responsibilities": [
            "Generate 701 synthetic records",
            "Maintain statistical distributions",
            "Data augmentation validation"
        ],
        "phase_focus": "Phase 1 & 3"
    }
}

# PROJECT TIMELINE (7 weeks)
PROJECT_TIMELINE = {
    "Week 1": {
        "phase": "Phase 1 - Data Collection & Setup",
        "deliverables": [
            "UCI dataset downloaded and validated",
            "Project structure created",
            "Data dictionary completed",
            "Team roles confirmed"
        ],
        "responsible": ["Cameron", "Jose"]
    },
    "Week 2": {
        "phase": "Phase 2 - Exploratory Data Analysis",
        "deliverables": [
            "Statistical summaries generated",
            "Data visualization suite",
            "Initial correlation analysis",
            "EDA notebook completed"
        ],
        "responsible": ["Nathan", "Nasiru"]
    },
    "Week 3": {
        "phase": "Phase 3 - Feature Engineering",
        "deliverables": [
            "Feature selection using Chi-square",
            "Data preprocessing pipeline", 
            "Synthetic data generation (701 records)",
            "Feature engineering notebook"
        ],
        "responsible": ["Julian", "Jose"]
    },
    "Week 4": {
        "phase": "Phase 4 - Model Development",
        "deliverables": [
            "Decision tree classifier implementation",
            "Hyperparameter tuning",
            "Cross-validation setup",
            "Model training notebook"
        ],
        "responsible": ["Juan", "Julian"]
    },
    "Week 5": {
        "phase": "Phase 5 - Model Evaluation",
        "deliverables": [
            "ROC curve analysis",
            "Bootstrap validation",
            "Performance metrics calculation",
            "Model evaluation notebook"
        ],
        "responsible": ["Nathan", "Juan"]
    },
    "Week 6": {
        "phase": "Phase 6 - Final Testing & Documentation",
        "deliverables": [
            "Final model testing",
            "Decision tree visualization",
            "Performance comparison with hospital baseline",
            "Final presentation preparation"
        ],
        "responsible": ["Nasiru", "Cameron"]
    },
    "Week 7": {
        "phase": "Project Finalization",
        "deliverables": [
            "Final report completion",
            "Presentation delivery",
            "Code documentation",
            "Project submission"
        ],
        "responsible": ["All team members"]
    }
}

# PROJECT GOALS AND SUCCESS METRICS
PROJECT_GOALS = {
    "primary_goal": "Achieve ≥85% diagnostic accuracy for heart failure prediction",
    "baseline_comparison": "Hospital accuracy: 83.9%",
    "minimum_improvement": "1% improvement (clinically meaningful)",
    "confidence_level": "95%",
    "dataset_size": "1000 records (299 real + 701 synthetic)",
    "evaluation_metrics": [
        "Accuracy (≥85%)",
        "Precision",
        "Recall (Sensitivity)", 
        "AUC-ROC",
        "k-Fold Cross-Validation"
    ]
}

# COMMUNICATION PROTOCOLS
COMMUNICATION = {
    "weekly_meetings": "Every Monday 7:00 PM EST",
    "progress_updates": "Wednesday check-ins via Discord",
    "code_reviews": "Before merging any phase completion", 
    "documentation": "All code must include docstrings and comments",
    "git_workflow": "Feature branches for each phase, main branch for stable releases"
}

def print_team_overview():
    """Print formatted team overview and responsibilities."""
    print("="*80)
    print(f"TEAM: {TEAM_NAME}")
    print("="*80)
    
    for member, info in TEAM_MEMBERS.items():
        print(f"\n{member} - {info['role']}")
        print(f"Phase Focus: {info['phase_focus']}")
        print("Responsibilities:")
        for resp in info['responsibilities']:
            print(f"  • {resp}")
    
    print("\n" + "="*80)
    print("PROJECT SUCCESS CRITERIA")
    print("="*80)
    print(f"Target Accuracy: {PROJECT_GOALS['primary_goal']}")
    print(f"Baseline to Beat: {PROJECT_GOALS['baseline_comparison']}")
    print(f"Confidence Level: {PROJECT_GOALS['confidence_level']}")
    print(f"Dataset Size: {PROJECT_GOALS['dataset_size']}")

def print_timeline():
    """Print detailed project timeline."""
    print("="*80)
    print("PROJECT TIMELINE (7 WEEKS)")
    print("="*80)
    
    for week, details in PROJECT_TIMELINE.items():
        print(f"\n{week}: {details['phase']}")
        print(f"Team Lead(s): {', '.join(details['responsible'])}")
        print("Deliverables:")
        for deliverable in details['deliverables']:
            print(f"  • {deliverable}")

if __name__ == "__main__":
    print_team_overview()
    print("\n")
    print_timeline()