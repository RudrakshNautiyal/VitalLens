# VitalLens

> An end-to-end diabetes risk prediction system with explainable AI, REST API deployment, and a live Streamlit dashboard.

---

## Problem statement

Diabetes affects over 537 million adults globally, yet a large proportion of cases go undiagnosed until complications arise. Early detection significantly improves patient outcomes, but access to clinical screening remains limited — especially in rural and low-income regions where laboratory tests and specialist visits are out of reach.

**VitalLens** addresses this gap by predicting an individual's risk of Type 2 diabetes from 8 routine health metrics. Crucially, it doesn't just output a number — it uses SHAP (SHapley Additive exPlanations) to show *which* health factors are driving the risk score, making predictions interpretable and actionable for both patients and practitioners.

---

## Demo

![Dashboard screenshot](assets/dashboard.png)

> Live input → risk score → SHAP waterfall explanation, all in one screen.

---

## Tech stack

| Layer | Tools |
|---|---|
| Data & EDA | Pandas, NumPy, Seaborn, Matplotlib |
| Modelling | Scikit-learn, PyTorch, Optuna |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Backend API | FastAPI, Pydantic |
| Testing | pytest |
| Frontend | Streamlit |
| Containerisation | Docker, Docker Compose |

---

## Project structure

```
vitallens/
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── eda_and_training.ipynb      # full EDA + model comparison
├── src/
│   ├── preprocess.py               # data cleaning, imputation, feature engineering
│   ├── train.py                    # model training + MLflow logging
│   ├── evaluate.py                 # ROC-AUC, CV, confusion matrix
│   └── explain.py                  # SHAP value generation
├── api/
│   ├── main.py                     # FastAPI app
│   ├── schemas.py                  # Pydantic request/response models
│   └── model_loader.py             # load saved model + scaler
├── app/
│   └── streamlit_app.py            # Streamlit dashboard
├── tests/
│   └── test_api.py                 # pytest test suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Dataset

**PIMA Indians Diabetes Dataset** — UCI / Kaggle  
768 records · 8 features · binary classification (diabetic / non-diabetic)

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Genetic diabetes likelihood score |
| Age | Age in years |

---

## Data preprocessing pipeline

Key challenges and how they are handled:

**1. Impossible zeros** — Glucose, BloodPressure, BMI, Insulin, and SkinThickness contain biological zeros that are actually missing values. These are replaced with `NaN` and imputed using the per-class median (diabetic vs non-diabetic), which is more accurate than a global median.

**2. Feature engineering** — Three derived features are added:
- `BMI_Age` — interaction term between BMI and Age
- `Glucose_Insulin` — metabolic ratio (glucose / insulin + 1)
- `High_BP` — binary flag for diastolic BP above 80 mmHg

**3. Class imbalance** — The dataset is ~65% non-diabetic. SMOTE (Synthetic Minority Oversampling Technique) is applied on the training set only to balance classes without leaking test data.

**4. Scaling** — `StandardScaler` is fit on the training set and applied to both train and test sets. The scaler is saved alongside the model to ensure consistent inference.

---

## Models trained

Three models are trained, tracked in MLflow, and compared:

| Model | ROC-AUC | Recall | F1-score |
|---|---|---|---|
| Logistic Regression | 0.83 | 0.76 | 0.74 |
| Random Forest | 0.89 | 0.81 | 0.79 |
| PyTorch MLP | 0.88 | 0.80 | 0.78 |

> Random Forest selected as the production model. Hyperparameters tuned with Optuna.

**Why recall over accuracy?** A false negative — telling a diabetic patient they are healthy — is clinically worse than a false positive. The model and threshold are optimised accordingly.

---

## Explainability

SHAP waterfall charts are generated per prediction, showing the contribution of each feature to the final risk score. This makes every prediction auditable and explainable in plain language.

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_input)
shap.waterfall_plot(shap_values[0])
```

---

## API reference

Run the API locally:

```bash
uvicorn api.main:app --reload
```

**POST** `/predict`

Request:
```json
{
  "pregnancies": 2,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree": 0.627,
  "age": 50
}
```

Response:
```json
{
  "risk_score": 0.83,
  "prediction": "High risk",
  "top_factors": ["Glucose", "BMI", "Age"]
}
```

**GET** `/health` — returns API status

---

## Running with Docker

```bash
git clone https://github.com/yourusername/vitallens.git
cd vitallens
docker-compose up --build
```

- API available at `http://localhost:8000`
- Streamlit dashboard at `http://localhost:8501`
- MLflow UI at `http://localhost:5000`

---

## Running tests

```bash
pytest tests/ -v
```

Tests cover: input validation, prediction endpoint, edge cases (zero values, out-of-range inputs), and response schema.

---

## Experiment tracking

All training runs are logged to MLflow — parameters, metrics, and model artifacts. To view:

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to compare runs.

---

## Planned extensions

- Heart disease risk model (Cleveland Heart Disease dataset)
- Kidney disease risk model (UCI CKD dataset)
- User authentication for the dashboard
- Model drift monitoring with Evidently AI

---

## Results summary

- ROC-AUC: **0.89**
- Recall: **0.81**
- F1-score: **0.79**
- Dataset size: **768 records**
- Features used: **11** (8 original + 3 engineered)

---

## Author

**Rudraksh Nautiyal**  
B.Tech CSE · Jaypee University of Information Technology  
[LinkedIn](https://linkedin.com) · [GitHub](https://github.com) · [GeeksForGeeks](https://geeksforgeeks.org)

---

## License

MIT License — free to use, modify, and distribute with attribution.
