# Diabetes Dataset Analysis Project

## Project Overview
This project explores the Diabetes dataset to uncover insights and relationships between various health metrics and diabetes outcomes. The analysis is conducted in Python using popular data science libraries.

## Dataset Information
- **Source**: `diabetes.csv`
- **Number of entries**: 768
- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (0: Non-Diabetic, 1: Diabetic)

## Objectives
1. Explore the distribution of diabetic vs. non-diabetic patients.
2. Analyze the relationship between glucose levels and diabetes outcomes.
3. Investigate the role of BMI in diabetes diagnosis.

## Key Findings

### 1. Distribution of Diabetic vs. Non-Diabetic Patients
- **Non-Diabetic (Outcome = 0)**: 500 patients
- **Diabetic (Outcome = 1)**: 268 patients

A visual representation using a count plot shows the imbalance between the two classes.

### 2. Relationship Between Glucose Levels and Outcome
- **Average glucose level for non-diabetic patients**: ~109.98 mg/dL
- **Average glucose level for diabetic patients**: ~141.26 mg/dL

A box plot illustrates that glucose levels are significantly higher in diabetic patients.

### 3. Role of BMI in Diabetes
- **Average BMI for non-diabetic patients**: ~30.30
- **Average BMI for diabetic patients**: ~35.14

A box plot confirms that BMI tends to be higher in diabetic patients.

## Tools and Libraries Used
- `pandas`: Data manipulation and analysis
- `seaborn`: Data visualization
- `matplotlib`: Plotting graphs
- google colab Notebook: Interactive environment for code execution

## How to Run the Project
1. Clone or download the project files.
2. Ensure you have Python installed along with the required libraries.
3. Open `Project_2.ipynb` in Jupyter\colab Notebook.
4. Run the cells sequentially to reproduce the analysis.

## Future Work
- Perform more advanced statistical tests.
- Build a predictive model to classify diabetes outcomes.
- Explore other features and their correlations.
