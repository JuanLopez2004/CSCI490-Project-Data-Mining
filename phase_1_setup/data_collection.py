"""
Phase 1: Data Collection and Initial Setup
Team: Goblins Hiding in Vents (Cameron, Julian, Juan, Nathan, Nasiru, Jose)
Goal: Collect UCI heart failure data and prepare synthetic data generation

This module handles initial data collection and setup for the heart failure prediction project.
Target: 299 UCI records + 701 synthetic records = 1000 total samples
"""

import pandas as pd
import numpy as np
import requests
import os
import logging
from typing import Tuple, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Handles data collection from UCI repository and initial data validation.
    
    Responsibilities:
    - Download UCI Heart Failure Clinical Records dataset
    - Validate data integrity and structure
    - Prepare data directories and file organization
    - Initial data quality assessment
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize data collector with specified data directory.
        
        Args:
            data_dir: Path to store collected data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset metadata
        self.uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
        self.expected_features = [
            'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
            'sex', 'smoking', 'time', 'DEATH_EVENT'
        ]
        self.expected_rows = 299
        
    def download_uci_dataset(self) -> bool:
        """
        Download the UCI Heart Failure Clinical Records dataset.
        
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            logger.info("Downloading UCI Heart Failure Clinical Records dataset...")
            
            # Download the dataset
            response = requests.get(self.uci_url, timeout=30)
            response.raise_for_status()
            
            # Save to data directory
            filepath = self.data_dir / "heart_failure_clinical_records_dataset.csv"
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Dataset downloaded successfully to {filepath}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to download dataset: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            return False
    
    def validate_dataset_structure(self) -> Dict[str, Any]:
        """
        Validate the downloaded dataset structure and content.
        
        Returns:
            Dict containing validation results and basic statistics
        """
        try:
            filepath = self.data_dir / "heart_failure_clinical_records_dataset.csv"
            
            if not filepath.exists():
                return {"valid": False, "error": "Dataset file not found"}
            
            # Load dataset
            df = pd.read_csv(filepath)
            
            # Validate structure
            validation_results = {
                "valid": True,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "expected_rows": self.expected_rows,
                "expected_columns": len(self.expected_features),
                "missing_features": [],
                "extra_features": [],
                "missing_values": {},
                "data_types": {},
                "basic_stats": {}
            }
            
            # Check for missing or extra features
            actual_features = set(df.columns)
            expected_features = set(self.expected_features)
            
            validation_results["missing_features"] = list(expected_features - actual_features)
            validation_results["extra_features"] = list(actual_features - expected_features)
            
            # Check data types and missing values
            for col in df.columns:
                validation_results["data_types"][col] = str(df[col].dtype)
                validation_results["missing_values"][col] = df[col].isnull().sum()
            
            # Basic statistics for numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            validation_results["basic_stats"] = df[numerical_cols].describe().to_dict()
            
            # Overall validation status
            if (validation_results["missing_features"] or 
                len(df) != self.expected_rows or
                any(validation_results["missing_values"].values())):
                validation_results["valid"] = False
                
            logger.info(f"Dataset validation completed. Valid: {validation_results['valid']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during dataset validation: {e}")
            return {"valid": False, "error": str(e)}
    
    def create_data_dictionary(self) -> bool:
        """
        Create a data dictionary explaining all features in the dataset.
        
        Returns:
            bool: True if dictionary created successfully
        """
        try:
            data_dictionary = {
                "age": {
                    "description": "Age of the patient",
                    "type": "Continuous",
                    "unit": "Years",
                    "range": "40-95"
                },
                "anaemia": {
                    "description": "Decrease of red blood cells or hemoglobin",
                    "type": "Binary",
                    "values": "0 = No, 1 = Yes"
                },
                "creatinine_phosphokinase": {
                    "description": "Level of the CPK enzyme in the blood",
                    "type": "Continuous", 
                    "unit": "mcg/L",
                    "range": "23-7861"
                },
                "diabetes": {
                    "description": "If the patient has diabetes",
                    "type": "Binary",
                    "values": "0 = No, 1 = Yes"
                },
                "ejection_fraction": {
                    "description": "Percentage of blood leaving the heart at each contraction",
                    "type": "Continuous",
                    "unit": "Percentage",
                    "range": "14-80"
                },
                "high_blood_pressure": {
                    "description": "If the patient has hypertension",
                    "type": "Binary", 
                    "values": "0 = No, 1 = Yes"
                },
                "platelets": {
                    "description": "Platelets in the blood",
                    "type": "Continuous",
                    "unit": "kiloplatelets/mL",
                    "range": "25100-850000"
                },
                "serum_creatinine": {
                    "description": "Level of serum creatinine in the blood",
                    "type": "Continuous",
                    "unit": "mg/dL",
                    "range": "0.5-9.4"
                },
                "serum_sodium": {
                    "description": "Level of serum sodium in the blood",
                    "type": "Continuous", 
                    "unit": "mEq/L",
                    "range": "113-148"
                },
                "sex": {
                    "description": "Woman or man",
                    "type": "Binary",
                    "values": "0 = Woman, 1 = Man"
                },
                "smoking": {
                    "description": "If the patient smokes",
                    "type": "Binary",
                    "values": "0 = No, 1 = Yes"
                },
                "time": {
                    "description": "Follow-up period",
                    "type": "Continuous",
                    "unit": "Days",
                    "range": "4-285"
                },
                "DEATH_EVENT": {
                    "description": "If the patient deceased during the follow-up period",
                    "type": "Binary",
                    "values": "0 = No, 1 = Yes",
                    "note": "TARGET VARIABLE"
                }
            }
            
            # Save as JSON and CSV
            import json
            
            # JSON format
            with open(self.data_dir / "data_dictionary.json", 'w') as f:
                json.dump(data_dictionary, f, indent=2)
            
            # CSV format for easy viewing
            dict_df = pd.DataFrame.from_dict(data_dictionary, orient='index')
            dict_df.to_csv(self.data_dir / "data_dictionary.csv")
            
            logger.info("Data dictionary created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating data dictionary: {e}")
            return False
    
    def setup_project_structure(self) -> bool:
        """
        Create the complete project directory structure.
        
        Returns:
            bool: True if structure created successfully
        """
        try:
            directories = [
                "data/raw",
                "data/processed", 
                "data/synthetic",
                "notebooks",
                "src",
                "models",
                "reports",
                "phase_1_setup",
                "phase_2_eda", 
                "phase_3_features",
                "phase_4_modeling",
                "phase_5_evaluation",
                "phase_6_final"
            ]
            
            for directory in directories:
                dir_path = Path("..") / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                
            logger.info("Project directory structure created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating project structure: {e}")
            return False


def main():
    """
    Main function to execute Phase 1 data collection workflow.
    
    Team members should run this to:
    1. Download UCI dataset
    2. Validate data structure  
    3. Create data dictionary
    4. Set up project directories
    """
    print("="*60)
    print("PHASE 1: DATA COLLECTION AND SETUP")
    print("Team: Goblins Hiding in Vents")
    print("Goal: Collect and validate heart failure dataset")
    print("="*60)
    
    # Initialize data collector
    collector = DataCollector()
    
    # Step 1: Download dataset
    print("\n1. Downloading UCI Heart Failure dataset...")
    if collector.download_uci_dataset():
        print("✅ Dataset downloaded successfully")
    else:
        print("❌ Dataset download failed")
        return
    
    # Step 2: Validate dataset
    print("\n2. Validating dataset structure...")
    validation = collector.validate_dataset_structure()
    
    if validation["valid"]:
        print("✅ Dataset validation passed")
        print(f"   Rows: {validation['total_rows']}")
        print(f"   Columns: {validation['total_columns']}")
    else:
        print("❌ Dataset validation failed")
        if "error" in validation:
            print(f"   Error: {validation['error']}")
        if validation.get("missing_features"):
            print(f"   Missing features: {validation['missing_features']}")
    
    # Step 3: Create data dictionary
    print("\n3. Creating data dictionary...")
    if collector.create_data_dictionary():
        print("✅ Data dictionary created")
    else:
        print("❌ Data dictionary creation failed")
    
    # Step 4: Setup project structure
    print("\n4. Setting up project structure...")
    if collector.setup_project_structure():
        print("✅ Project structure created")
    else:
        print("❌ Project structure setup failed")
    
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE!")
    print("Next: Proceed to Phase 2 - Exploratory Data Analysis")
    print("="*60)


if __name__ == "__main__":
    main()