# Diabetes Prediction Model

## Project Overview
This project focuses on developing a machine learning model to predict diabetes using patient health data. The project is implemented in a Google Colab environment and explores various aspects of data analysis, visualization, and preliminary modeling.

## Dataset
The dataset used is `diabetes.csv`, which contains health metrics from 768 patients. Key features include:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0: Non-Diabetic, 1: Diabetic)

## Key Findings
### Data Exploration
- The dataset contains 768 entries with no null values
- Class distribution: 500 non-diabetic vs 268 diabetic patients

### Key Insights
1. **Glucose Levels**: Diabetic patients show significantly higher average glucose levels (141.26) compared to non-diabetic patients (109.98)
2. **BMI**: Diabetic patients have higher average BMI (35.14) compared to non-diabetic patients (30.30)
3. **Visual Analysis**: Box plots show clear separations in glucose levels and BMI distributions between diabetic and non-diabetic groups

## Technologies Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Google Colab

## Getting Started

### Prerequisites
- Google account for accessing Colab
- Basic understanding of Python and machine learning concepts

### Installation
1. Clone the repository:
```bash
git clone https://github.com/0Maxbon0/GTC-ML-Project-2---Diabetes-Prediction-Model.git
