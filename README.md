# Heart Failure Prediction – Goblins Hiding in Vents

## Setup

### Prerequisites
- Python 3.12.4
- Git

### Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JuanLopez2004/csci490.git
   cd csci490
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   
   **Windows (PowerShell):**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   
   **Windows (Command Prompt):**
   ```cmd
   .venv\Scripts\activate.bat
   ```
   
   **macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

---

## Project Overview
Heart disease is the leading cause of death for men, women, and most racial and ethnic groups. Diagnosing heart failure (HF) effectively is crucial, as misdiagnosis rates range from 16% in hospitals to 68% when patients are referred to specialists. This project aims to develop a machine learning model that predicts heart failure with higher accuracy than current hospital diagnoses.

**Goal:** Achieve at least 85% accuracy with 95% confidence in predicting heart failure, improving upon the typical hospital success rate of 83.9%.

---

## Team
- Cameron  
- Julian  
- Juan

## Citations
https://docs.google.com/document/d/1_SOi75va9zlUbRK0rf99x6agDCygg5cdAIQ4XwGuLLQ/edit?usp=sharing

## Proposal
https://docs.google.com/document/d/16qZ9RXL_Jhs3Sdk-vMXuFFfAF-A4F4yCTPOypt5yivo/edit?usp=sharing

Team Name: *Goblins Hiding in Vents*

---

## Dataset
- Source: [UCI Machine Learning Repository – Heart Failure Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)
- Number of patients: 299
- Number of features: 13
- Features include: age, sex, blood pressure, creatinine, ejection fraction, smoking status, and more.
- Target: `DEATH_EVENT` (1 = patient died, 0 = patient survived)

**Dataset Highlights:**  
- Modern dataset (2020) compared to older datasets like Cleveland 1988.  
- Includes clinical and demographic attributes for risk prediction.

---

## Methods
- **Data Cleaning:** Handle missing values, encode categorical features, normalize numeric features.  
- **Exploratory Data Analysis (EDA):** Descriptive statistics, feature distributions, correlation analysis.  
- **Feature Engineering:** Derived features, Chi-Square tests, final feature selection.  
- **Modeling:** Decision Tree Classifier as baseline; hyperparameter tuning and cross-validation to improve performance.  
- **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix, and confidence intervals.

---

## Project Workflow
1. **Phase 1 – Setup & Research:** Repository setup, literature review, project objectives.  
2. **Phase 2 – Data Cleaning:** Handling missing values, encoding, scaling.  
3. **Phase 3 – Exploratory Data Analysis:** Feature exploration, correlations, visualizations.  
4. **Phase 4 – Feature Engineering:** Derived features, Chi-Square analysis, final feature selection.  
5. **Phase 5 – Modeling:** Train Decision Tree, evaluate, save model.  
6. **Phase 6 – Accuracy Improvement:** Hyperparameter tuning, cross-validation, model comparison.  
7. **Phase 7 – Checkpoint Report:** Compile findings, visualizations, and metrics into a report.

---

## Jupyter Notebooks
The following notebooks are part of this repository:

---

## Checkpoint Deliverables
- 2–4 page PDF report including:

---

## Expected Results
- Identify heart failure with **≥85% accuracy**.  
- Provide interpretable Decision Tree model with **feature importance analysis**.  
- Improve prediction reliability compared to hospital diagnoses.

---

## License
This project is for academic purposes. Dataset usage follows UCI Machine Learning Repository guidelines.

---
## Contact
For questions or collaboration, reach out to the team via GitHub Issues or email.
