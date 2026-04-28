# Data Mining Analysis of Deaths Due to COVID-19

Analysis of **37,351 Malaysian COVID-19 death records** spanning **2020–2024** across **16 states**, sourced from Malaysia's official public health linelist dataset.

## Project Overview

This project applies three data mining techniques to uncover mortality risk patterns across demographic, clinical, and vaccination-related variables. The full pipeline covers data cleaning, feature engineering, exploratory data analysis (EDA), machine learning modelling, and an interactive Streamlit dashboard for dynamic exploration.

## Key Results

- **Logistic Regression** achieved **79% accuracy** in classifying high-risk (brought-in-dead) patients
- **K-Means Clustering** identified **4 distinct patient risk segments** based on age, comorbidity, vaccination status, and time-to-death
- **Association Rule Mining (ARM)** surfaced co-occurring risk factors across demographic and clinical variables
- **5 engineered features**: vaccination status (unvaccinated/partial/full/booster), age group bins, days from positive test to death, days since last dose, and comorbidity flags

## Dashboard Sections

The interactive Streamlit dashboard includes:

- **Overview** — Dataset preview with global sidebar filters (state, age group, gender)
- **Demographic Analysis** — Deaths by state, age group, and gender
- **Vaccination & Health Factors** — Deaths by vaccination status, comorbidity, and vaccine brand
- **Filtered Death Records** — Dynamic table based on selected filters
- **Data Mining Insights** — Risk prediction factors (classification), association rules (ARM), and patient risk profiles (clustering)

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Streamlit · Plotly · Pickle

## Project Structure

```
├── Code/
│   ├── Dashboard/
│   │   ├── app.py                      # Streamlit dashboard
│   │   ├── covid_deaths_cleaned.csv    # Cleaned dataset (dashboard input)
│   │   ├── cluster_profiles.csv        # K-Means cluster results
│   │   ├── arm_rules.csv               # Association rules output
│   │   ├── log_reg_model.pkl           # Trained logistic regression model
│   │   └── background.png              # Dashboard background image
│   ├── data/
│   │   └── covid_deaths_linelist.csv   # Raw dataset
│   ├── notebooks/
│   │   ├── 01_cleaning_features.ipynb  # Data cleaning & feature engineering
│   │   ├── 02_eda_visuals.ipynb        # Exploratory data analysis
│   │   └── 03_mining_models.ipynb      # ML models & data mining
│   └── outputs/
│       ├── covid_deaths_cleaned.csv    # Cleaned dataset
│       ├── cluster_profiles.csv        # Clustering output
│       ├── arm_rules.csv               # ARM output
│       └── figures/                    # EDA charts (PNG)
├── Report.pdf                          # Full project report
└── README.md
```

## How to Run the Dashboard

**Requirements:** Python 3.9 or above

**Install dependencies:**

```bash
pip install streamlit pandas plotly
```

**Run the app:**

```bash
cd Code/Dashboard
streamlit run app.py
```

## Contributors
- Santoshkumar A/L K Munusamy
- Arvind A/L Gopalakrishna Thevar(https://github.com/Arvinz01)

The dashboard will open automatically in your web browser.
