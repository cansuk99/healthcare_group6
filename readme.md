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

├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── readme.md                    # Project documentation

├── data/
│   ├── preprocessing-codes-mapping/   # Codebooks and preprocessing mappings
│   ├── processed/                     # Cleaned and transformed datasets
│   ├── raw/                           # Original raw data
│   ├── selected-features/             # Subsets of features after selection
│   ├── exploratory_analysis/          # EDA outputs and summaries
│   ├── figures/                       # Generated plots and figures
│   ├── models/                        # Saved ML models
│   └── reports/                       # Evaluation reports and logs

├── ann/                               # Model training scripts
│   ├── step_1_data_cleaning.py
│   ├── step_2_feature_selection.py
│   ├── step_3_models.py
│   ├── step_4_ANN.py
│   ├── step_5_Logistic_Regression.py
│   └── step_6_Random_forest.py

└── src/                               # Core analysis pipeline
    ├── 01_data_exploration.py
    ├── 02_data_cleaning.py
    ├── 03_data_analysis.py
    ├── 04_feature_selection.py
    └── 05_models.py


data/ → all raw, processed, and analytical datasets

src/ → main analysis pipeline for EDA, cleaning, and modeling

ann/ → ANN model 

reports/ → stores model outputs, metrics, and visual summaries

exploratory analysis/ → ipynb and py files for exploration

requirements.txt → dependencies

