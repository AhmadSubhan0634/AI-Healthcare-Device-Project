# AI Healthcare Device: Smart Symptom Triage and Health Education Chatbot
 
A machine learning project that takes a patient's reported symptoms and predicts **how urgently they need care** (triage level) and **which disease is most likely** (disease classification). The goal is to support a consumer-facing symptom checker that points people toward the right level of care: emergency services, an urgent visit, a routine appointment, or self-care at home.
 
> **Disclaimer:** This is an academic project trained on **synthetic data**. It is not a medical device and must not be used for real diagnosis or treatment decisions. If you think you are having a medical emergency, contact your local emergency services.
 
---
 
## Table of Contents
 
- [Overview](#overview)
- [Dataset](#dataset)
- [Triage Levels](#triage-levels)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Tech Stack](#tech-stack)
- [Author](#author)
---
 
## Overview
 
The project covers the full machine learning workflow in a single Jupyter notebook:
 
1. Load and validate a preprocessed symptom dataset.
2. Explore class balance, symptom prevalence, and triage levels by disease.
3. Train a multi-class **triage classifier** and a multi-class **disease classifier**.
4. Compare Random Forest, Logistic Regression, and a Neural Network (MLP) using accuracy, precision, recall, and F1.
5. Add a regression view of urgency (MAE, RMSE, R²) and an unsupervised view of symptom structure (KMeans and silhouette score).
6. Save trained models and encoders so they can be loaded into a chatbot or app.
## Dataset
 
| Property | Value |
| --- | --- |
| Records | 100,000 synthetic patient records |
| Input features | 174 binary symptom features (0 = absent, 1 = present) |
| Common symptoms | 135 (`symptom_*` columns) |
| Rare symptoms | 39 (`rare_symptom_*` columns) |
| Disease classes | 46 |
| Triage classes | 4 |
| Missing values | 0 |
| Duplicate rows | 0 |
 
The notebook expects a CSV named `Final(processed dataset).csv` with a `disease` column, a `triage_level` column, and the 174 symptom columns. **The dataset file is not included in this repository.**
 
**Triage class distribution**
 
| Triage level | Records |
| --- | --- |
| Urgent | 34,797 |
| Non-urgent | 33,012 |
| Self-care | 24,410 |
| Emergency | 7,781 |
 
The data is imbalanced: Emergency is the smallest class, and a few diseases (Asthma, Hyperthyroidism, Irritable Bowel Syndrome) dominate the disease labels. Several rare diseases have only a handful of records.
 
**Diseases covered (46):** Acute Gastroenteritis, Amyotrophic Lateral Sclerosis, Asthma, COPD, COVID-19, Chronic Kidney Disease, Chronic Liver Disease, Cirrhosis, Common Cold, Coronary Artery Disease, Crohn's Disease, Cystic Fibrosis, Dengue, Depression, Ehlers-Danlos Syndrome, Epilepsy, Fanconi Anemia, Flu, Gastritis, Gaucher Disease, Generalized Anxiety Disorder, Hypertension, Hyperthyroidism, Hypothyroidism, Inflammatory Bowel Disease, Irritable Bowel Syndrome, Marfan Syndrome, Metabolic Syndrome, Migraine, Multiple Sclerosis, Niemann-Pick Disease, Obesity, Osteoarthritis, Osteopetrosis, Peptic Ulcer Disease, Pneumonia, Polycystic Ovary Syndrome, Psoriasis, Spinal Muscular Atrophy, Stroke, Systemic Lupus Erythematosus, Tuberculosis, Type 2 Diabetes, Typhoid Fever, Ulcerative Colitis, Urinary Tract Infection.
 
## Triage Levels
 
The four-level urgency system maps a prediction to a recommended care action:
 
| Level | Meaning |
| --- | --- |
| **Emergency** | Needs immediate emergency care |
| **Urgent** | Should be seen by a clinician soon |
| **Non-urgent** | Can be handled with a routine appointment |
| **Self-care** | Can be managed at home with health education |
 
## Models
 
| Model | Task | Notes |
| --- | --- | --- |
| Random Forest (baseline) | Triage | 100 trees, max depth 15, trained on the full training set |
| Random Forest (tuned) | Triage | 150 trees, max depth 20, min samples split 5, min samples leaf 2 |
| Logistic Regression | Triage | Features standardized with `StandardScaler`, 500 iterations |
| Neural Network (MLP) | Triage | Two hidden layers (128, 64), ReLU, early stopping, trained on the first 40,000 training rows |
| Random Forest | Disease | 50 trees, max depth 20, trained on a 30,000-row sample for speed |
| Random Forest Regressor | Urgency score | Triage encoded as 0 to 3, 50 trees, trained on a 20,000-row sample |
| MiniBatchKMeans (k = 4) | Clustering | Run on a 10,000-row sample, visualized with PCA |
 
All data is split **80/20** (random state 42, stratified by triage level for the classifiers).
 
## Results
 
### Classification (weighted averages on the 20,000-row test set)
 
| Model | Task | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Random Forest (baseline) | Triage | 0.5282 | 0.5655 | 0.5282 | 0.4840 |
| Logistic Regression | Triage | 0.5209 | 0.5139 | 0.5209 | 0.5105 |
| Random Forest (tuned) | Triage | 0.5475 | 0.5859 | 0.5475 | 0.5099 |
| **Neural Network (MLP)** | Triage | **0.5904** | 0.5890 | 0.5904 | **0.5891** |
| Random Forest | Disease (46 classes) | 0.3791 | 0.3891 | 0.3791 | 0.3288 |
 
The **Neural Network performed best** for triage, reaching about 59% accuracy and 0.589 weighted F1. It was also the only model with a reasonable balance on the Emergency class (precision 0.49, recall 0.44, F1 0.47). The Random Forest models had higher precision on Emergency but missed almost all of those cases (recall of 0.01 for the baseline and 0.03 for the tuned model).
 
### Regression (Random Forest on numeric triage)
 
| Metric | Value |
| --- | --- |
| MAE | 0.7567 |
| RMSE | 0.8929 |
| R² | 0.1857 |
 
### Clustering (KMeans, k = 4)
 
| Metric | Value |
| --- | --- |
| Silhouette score | 0.0720 |
 
The low silhouette score is expected: many diseases share overlapping symptoms, so symptoms alone do not form clean clusters.
 
## Project Structure
 
```
AI-Healthcare-Device-Project/
├── Smart_Symptom_Triage_Chatbot.ipynb   # Full notebook: EDA, modeling, evaluation
├── Final(processed dataset).csv         # Dataset (add this yourself, not included)
├── models/                              # Created when the notebook runs
│   ├── le_disease.pkl                   # Disease label encoder
│   ├── le_triage.pkl                    # Triage label encoder
│   ├── symptom_cols.pkl                 # Ordered list of the 174 symptom columns
│   ├── rf_triage.pkl                    # Baseline Random Forest (triage)
│   ├── rf_triage_tuned.pkl              # Tuned Random Forest (triage)
│   ├── rf_disease.pkl                   # Random Forest (disease)
│   ├── lr_triage.pkl                    # Logistic Regression (triage)
│   ├── scaler.pkl                       # Scaler for Logistic Regression
│   ├── mlp_triage.pkl                   # Neural Network (triage)
│   └── nn_scaler.pkl                    # Scaler for the Neural Network
└── plots/                               # Created when the notebook runs (12 figures)
```
 
## Getting Started
 
### Prerequisites
 
- Python 3.9 or newer
- Jupyter Notebook or JupyterLab
### Installation
 
```bash
git clone https://github.com/AhmadSubhan0634/AI-Healthcare-Device-Project.git
cd AI-Healthcare-Device-Project
 
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```
 
### Run the notebook
 
1. Place `Final(processed dataset).csv` in the project folder.
2. In the data-loading cell, update the file path to match your machine. The notebook currently changes into a Windows Desktop folder, so replace that with your own path (or remove those two lines if the CSV sits next to the notebook).
3. Launch Jupyter and run all cells in order:
```bash
jupyter notebook Smart_Symptom_Triage_Chatbot.ipynb
```
 
Running the notebook creates the `models/` and `plots/` folders automatically.
 
### Use a saved model
 
```python
import joblib, pickle
import numpy as np
 
mlp          = joblib.load("models/mlp_triage.pkl")
scaler       = joblib.load("models/nn_scaler.pkl")
le_triage    = joblib.load("models/le_triage.pkl")
symptom_cols = pickle.load(open("models/symptom_cols.pkl", "rb"))
 
# Build a 0/1 vector in the same column order used for training
selected = {"symptom_cough", "symptom_shortness_of_breath"}
x = np.array([[1 if col in selected else 0 for col in symptom_cols]])
 
pred = mlp.predict(scaler.transform(x))
print(le_triage.inverse_transform(pred)[0])
```
 
## Notebook Walkthrough
 
| Section | What it does |
| --- | --- |
| 1. Data Description | Loads the dataset, reports shape, feature counts, class balance, and data quality |
| 2. Data Preprocessing | Checks missing values and duplicates, builds the feature matrix, encodes labels, and splits train/test |
| 3. Visualizations | Triage distribution, disease distribution, top 20 symptoms, triage level by disease |
| 4. Modeling | Baseline Random Forest (triage and disease) and Logistic Regression |
| 5. Evaluation | Confusion matrix, top 20 feature importances, per-class precision, recall, and F1 |
| 6. Regression | Random Forest Regressor with MAE, RMSE, and R² |
| 7. Clustering | KMeans with silhouette score and a PCA 2D view |
| 8. Logistic Regression | Full evaluation with a confusion matrix |
| 9. Improved Models | Tuned Random Forest and Neural Network, with improvement and all-model comparison charts |
 
## Limitations
 
- **Synthetic data.** The records are not real patients, so results will not transfer to clinical settings.
- **Modest accuracy.** The best triage model reaches about 59% accuracy, and the disease classifier about 38%. Many diseases share symptoms, which limits what symptoms alone can predict.
- **Weak Emergency detection.** The Emergency class is small and hard to detect. The Random Forest models rarely flag it, which would be a serious problem in a real triage tool where missing an emergency is the costliest error.
- **Rare diseases.** Several diseases have very few records, so the disease classifier scores near zero on them.
- **Subsampled training.** To keep run times short, the disease model, MLP, regressor, and clustering use subsets of the data.
## Future Work
 
- Handle class imbalance (class weights, resampling, or threshold tuning) to improve Emergency recall.
- Train the neural network on the full dataset with longer training and hyperparameter search.
- Add patient context such as age, sex, and symptom duration alongside binary symptoms.
- Wrap the best model in a chatbot or web interface with health education content for each triage level.
- Validate on real, de-identified clinical data with clinician review.
## Tech Stack
 
Python · Jupyter · NumPy · Pandas · Matplotlib · Seaborn · Scikit-learn · Joblib
 
## Author
 
**Ahmed Subhan**
Computer Science student, FAST NUCES, Lahore
GitHub: [AhmadSubhan0634](https://github.com/AhmadSubhan0634) · LinkedIn: [ahmed-subhan-0820b5328](https://www.linkedin.com/in/ahmed-subhan-0820b5328)
 
