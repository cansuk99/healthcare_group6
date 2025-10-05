## Installation

Dataset used:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

To run the scripts in this repository, install all required Python packages with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Package List

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`

> All scripts are compatible with Python 3.7+.

## Project Structure

```
healthcare_group6/
├── data/
│   ├── raw/                        # Original dataset files
│   ├── processed/                  # Cleaned and transformed data
│   └── selected-feature/           # Selected features for modeling
├── reports/                        # Analysis reports and CSV summaries
├── figures/                        # Plots and visualizations
├── models/                         # Saved models and scalers
├── src/                            # Source code scripts
│   ├── 01_data_exploration.py
│   ├── 02_data_cleaning.py
│   ├── 03_data_analysis.py
│   ├── 04_feature_selection.py
│   └── 05_models.py
├── diabetes_type_exploratory_analysis.ipynb  # Exploratory analysis notebook
└── readme.md                       # Project documentation
```