# Student Performance Predictor

A Streamlit app that analyzes the UCI Student Performance dataset and predicts student performance categories (`Poor`, `Fair`, `Excellent`) using a Random Forest model.

## Project Overview

This project combines two datasets from the UCI repository: `student-mat.csv` and `student-por.csv`.
The app performs exploratory data analysis with charts and visualizations, trains a classifier on the combined dataset, and allows users to input student features to predict performance.

## What it does

- Loads and merges both datasets
- Preprocesses selected features and creates a target label based on final grade
- Displays exploratory visualizations for performance, demographics, study habits, and correlations
- Trains a Random Forest model on the dataset
- Predicts a student performance category from user-provided inputs

## How to use it

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. In the sidebar, provide student details such as:
   - school
   - age
   - address
   - mother/father education
   - mother/father job
   - reason for school
   - travel time, study time, failures
   - higher education desire, internet access, romantic relationship
   - alcohol use, health, G1 and G2 grades

4. The app will display:
   - predicted performance category
   - prediction probabilities for each category
   - model accuracy on holdout data
   - interactive EDA charts and a dataset sample

## Prediction logic

The app uses the students' attributes and exam grades to predict whether a student is likely to fall into one of three categories:

- `Poor` — final grade <= 11
- `Fair` — final grade 12 to 15
- `Excellent` — final grade > 15

The model is trained on the combined dataset and uses the input features to estimate the likelihood of each performance category.

## Dataset source

Cortez, Paulo. (2014). Student Performance. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.
