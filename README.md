# Diabetes Prediction Model: From Clinical Signal to Deployable Classifier

## The Architecture & The “Why”

Diabetes risk prediction is a binary classification problem with asymmetric cost: missing a likely-positive patient (false negative) is usually worse than flagging a non-diabetic patient for follow-up (false positive).  
This project uses the PIMA Indians Diabetes dataset (`diabetes.csv`) and builds a full notebook pipeline (`Project_2.ipynb`) that goes from raw tabular data to patient-level prediction.

At a high level, the architecture is:

1. **Ingest + profile** raw rows (`pandas`)
2. **Repair clinically invalid zeros** in physiological features
3. **Engineer interaction features** (ratio-based metabolic signals)
4. **Split + standardize** with train/test isolation
5. **Train and compare** Logistic Regression, Random Forest, and SVM
6. **Tune with GridSearchCV**
7. **Run interactive inference** from user-provided patient values

This design is intentionally practical: start with interpretable and classical baselines, then move to higher-capacity models only where they improve generalization on held-out data.

---

## Data Preprocessing & Feature Engineering

The core data issue is that several physiological columns contain `0` values that are not medically plausible in this context (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`).  
Instead of global imputation, the notebook imputes these as missing and fills them using **class-conditional medians** grouped by `Outcome`, preserving label-specific distribution shape.

Then the pipeline adds ratio features that encode nonlinear interactions:

- `Glucose_BMI_Ratio`
- `BloodPressure_BMI_Ratio`
- `Age_BMI_Ratio`
- `Insulin_Glucose_Ratio`

After feature construction:

- `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`
- `StandardScaler` fit on train only, applied to train/test

```python
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_features] = df[zero_features].replace(0, np.nan)
for feature in zero_features:
    med_pos = df[df['Outcome'] == 1][feature].median()
    med_neg = df[df['Outcome'] == 0][feature].median()
    df.loc[df[feature].isna() & (df['Outcome'] == 1), feature] = med_pos
    df.loc[df[feature].isna() & (df['Outcome'] == 0), feature] = med_neg
df['Glucose_BMI_Ratio'] = df['Glucose'] / df['BMI']
df['Insulin_Glucose_Ratio'] = df['Insulin'] / df['Glucose']
```

Why this matters: feature scale heterogeneity and invalid zeros can dominate optimization dynamics and obscure clinically meaningful signal if left untreated.

---

## Model Selection & Training

The notebook evaluates three standard binary classifiers:

- **Logistic Regression** (baseline linear boundary)
- **Random Forest Classifier** (nonlinear, robust to mixed interactions)
- **SVM (SVC)** (margin-based nonlinear decision boundary with kernels)

Each candidate is trained on the same split; then hyperparameters are tuned with `GridSearchCV`.  
In this repository, Random Forest is used as the final inference model and performs best among the tested options.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
```

Why Random Forest is a good fit here:

- Handles nonlinear interactions without heavy feature transforms
- Less sensitive to local feature scaling errors than margin-based models
- Captures threshold behavior common in clinical variables

---

## Evaluation Metrics (Why Accuracy Alone Is Insufficient)

The notebook reports accuracy during model comparison, but production-grade evaluation for medical classification should include:

- **Confusion Matrix**: raw counts of TP, TN, FP, FN
- **Precision**: among predicted positives, how many are truly positive
- **Recall (Sensitivity)**: among true positives, how many are detected
- **F1 Score**: harmonic mean of precision and recall

In this domain, **Recall is often more important than Precision**.  
A false negative can delay intervention for a high-risk patient; a false positive usually triggers additional tests, which is operationally cheaper than a missed case.

Recommended metric block to add to the evaluation stage:

```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(cm, precision, recall, f1)
```

---

## How to Run This Locally

```bash
git clone https://github.com/0Maxbon0/GTC-ML-Project-2---Diabetes-Prediction-Model.git
cd GTC-ML-Project-2---Diabetes-Prediction-Model
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This repo currently keeps training and inference in `Project_2.ipynb`. To run inference:

1. Open the notebook: `jupyter notebook Project_2.ipynb`
2. Run all preprocessing/training cells in order
3. Run the final `predict_diabetes()` cell
4. Enter patient values at the prompts to get class prediction and probability

---

## Engineering Notes

- The final notebook inference cell trains `RandomForestClassifier` directly on unscaled `X_train`; earlier model comparison uses scaled data. For production, lock one preprocessing contract and reuse it for both training and inference.
- Class-conditional imputation currently uses `Outcome` labels. That is acceptable for offline analysis but should be converted to train-only transformers for deployment to avoid leakage patterns in future pipelines.
- This model is an educational/engineering artifact, not a clinical decision system.
