# Heart Disease Risk Predictor

A machine learning model that predicts whether heart disease is present from 13 routine clinical measurements. It is deployed as an interactive Streamlit web app.

**Live app:** https://arush-major-project.streamlit.app/

> **Disclaimer:** This is a learning project trained on 270 patient records. It is not a medical device and its output is not medical advice. Always consult a qualified clinician.

## Results

| Model | 5-fold CV accuracy (train) | Test accuracy | Test F1 (Presence) |
|---|---|---|---|
| Logistic Regression (tuned) | 0.82 | 0.91 | 0.88 |
| **Ridge Classifier (tuned, final)** | 0.81 | **0.93** | **0.90** |

The test set holds 54 patients, so each misclassified patient moves test accuracy by about 2 points. Treat the CV score as the more stable estimate.

**Strongest predictors (Ridge coefficients):** number of vessels coloured by fluoroscopy, thallium stress-test result, chest-pain type, male sex and ST depression.

## How it works

1. **EDA:** distributions, class balance, a correlation heatmap, a pair plot and chest-pain-type vs outcome.
2. **Preprocessing:** a scikit-learn `ColumnTransformer` imputes and scales the numeric features and one-hot encodes sex. It is wrapped in a `Pipeline` together with the model, so the same transforms run at training and prediction time.
3. **Modelling:** Logistic Regression and Ridge Classifier, each tuned with `RandomizedSearchCV`.
4. **Deployment:** the full pipeline is saved with `joblib` (`Cardio_healthRiskPred.pkl`) and loaded by `app.py`. The app returns a prediction and personalised lifestyle tips.

## Dataset

270 patients and 14 columns from the classic *Statlog (Heart)* dataset.

| Feature | Description |
|---|---|
| Age | Age in years |
| Sex | 0 = female, 1 = male |
| Chest pain type | 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic |
| BP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FBS over 120 | Fasting blood sugar > 120 mg/dl (0/1) |
| EKG results | Resting ECG result (0–2) |
| Max HR | Maximum heart rate achieved |
| Exercise angina | Exercise-induced angina (0/1) |
| ST depression | ST depression induced by exercise relative to rest |
| Slope of ST | Slope of the peak exercise ST segment (1–3) |
| Number of vessels fluro | Major vessels coloured by fluoroscopy (0–3) |
| Thallium | Thallium stress-test result (3 = normal, 6 = fixed defect, 7 = reversible defect) |
| **Heart Disease** | **Target:** Presence / Absence |

## Run locally

```bash
git clone https://github.com/Arush70/Heart_Disease_Predictor-Project.git
cd Heart_Disease_Predictor-Project
pip install -r requirements.txt
streamlit run app.py
```

To retrain, open `Cardio_health_Risk.ipynb` and run all cells.

## Project structure

```
├── Cardio_health_Risk.ipynb     # EDA, preprocessing, training, tuning, evaluation
├── Heart_Disease_Prediction.csv # dataset
├── Cardio_healthRiskPred.pkl    # trained pipeline (preprocessing + Ridge Classifier)
├── app.py                       # Streamlit app
└── requirements.txt
```

## Tech stack

Python · pandas · scikit-learn · matplotlib / seaborn · Streamlit · joblib

## Author

**Arush Kumar Vishwakarma**, [GitHub](https://github.com/Arush70)
