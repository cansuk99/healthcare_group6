🏥 Predicting Hospital Readmissions for Diabetic Patients

This project investigates whether early hospital readmissions (within 30 days) for diabetic patients can be predicted based on clinical and demographic information collected from 130 US hospitals between 1999 and 2008.

The dataset, sourced from the UCI Machine Learning Repository, includes over 100,000 encounters with details about patient demographics, diagnoses, lab results, and medications.
The goal is to determine whether these features contain enough information to build a reliable predictive model.

## Installation

Dataset used:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

To run the scripts in this repository, install all required Python packages with:

###
```bash
pip install requirements.txt
```

**data/** → all raw, processed, and analytical datasets

**src/** → main analysis pipeline for EDA, cleaning, and modeling

**ann/** → ANN model. Model is using **all features**

**reports/** → stores model outputs, metrics, and visual summaries

**exploratory analysis/** → ipynb and py files for exploration
    ` feature-selection-interaction.ipynb ` - Logisitc Regression for finding statistically sigfnificant interaction features (mentioned in presentation)
    ` manual_model_implementation.py ` -- 3 Models with **manually selected features** based on feature-selection-interaction.ipynb (mentioned in presentation)


- Pipeline stages in src folder **(Automated feature selection)**:
  - Step 1: Data acquisition & initial exploration (src/01_data_exploration.py)
  - Step 2: Cleaning, ICD‑9 mapping, feature engineering (src/02_data_cleaning.py)
  - Step 3: Exploratory data analysis (src/03_data_analysis.py)
  - Step 4: Feature engineering & selection (src/04_feature_selection.py)
  - Step 5: Model training, evaluation, and export (src/05_models.py)
