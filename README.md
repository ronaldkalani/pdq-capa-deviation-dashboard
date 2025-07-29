# AI-Powered Pharmacovigilance: Real-Time CAPA Monitoring, Risk Prediction, and Signal Detection

## Project Title
**AI-Powered Pharmacovigilance: Real-Time CAPA Monitoring, Risk Prediction, and Signal Detection from FDA FAERS Adverse Event Reports**

---

## 🎯 Goal
To develop an AI/ML-enabled system that automates signal detection, identifies Corrective and Preventive Action (CAPA) candidates, flags treatment deviations, and predicts serious adverse drug reactions (ADRs) using structured FAERS datasets. The system supports real-time quality monitoring for pharmacovigilance teams and regulators.

---

## 👥 Intended Audience
- Pharmaceutical Quality Teams (PDQ, PV)
- Regulatory Compliance Officers (FDA, EMA, Health Canada)
- Clinical Safety Data Scientists
- Healthcare AI Researchers
- GxP Auditors & Risk Managers

---

## 🔄 Strategy & Pipeline Steps
1. **Data Acquisition**  
   Load FAERS Q4 2020 datasets (`DEMO`, `DRUG`, `REAC`, `THER`, `INDI`, `OUTC`).

2. **Data Cleaning & Validation**  
   Identify missing values, invalid therapy dates, and non-standardized drug names.

3. **Deviation Tracking (THER)**  
   Flag inconsistent therapy date entries and quantify errors.

4. **CAPA Flag Simulation (DRUG + REAC)**  
   Identify high-frequency drug-reaction pairs to simulate potential CAPA triggers.

5. **Mismatch Detection (INDI vs REAC)**  
   Highlight differences between drug indications and reported adverse reactions.

6. **Predictive Modeling (DEMO + OUTC)**  
   Train a model to predict serious outcomes (death) from demographic features.

7. **Streamlit Dashboard**  
   Deploy an interactive dashboard with metric cards and real-time updates.

---

## ⚠️ Challenges
- FAERS data is **highly imbalanced**, with serious outcomes underrepresented.
- Therapy timelines often contain **imprecise placeholder dates** (e.g., 1970-01-01).
- Need to reconcile **ambiguous drug names** and duplicate reports.
- **Lightweight front-end requirement** for big data visualization (Streamlit).

---

## ❓ Problem Statement
How can we leverage historical pharmacovigilance records to automate the detection of safety signals, simulate CAPA decisions, and predict the seriousness of adverse drug events in near real-time?

---

## 🗃️ Dataset
FDA FAERS Q4 2020 public datasets:
- `DEMO20Q4.txt` – Patient demographics
- `DRUG20Q4.txt` – Administered drugs
- `REAC20Q4.txt` – Adverse reactions
- `THER20Q4.txt` – Therapy start and end dates
- `INDI20Q4.txt` – Drug indications
- `OUTC20Q4.txt` – Patient outcomes (e.g., Death, Recovery)

---

## 🤖 Machine Learning Prediction & Outcomes
- **Model:** `RandomForestClassifier`
- **Features Used:** `age`, `sex`, `wt`
- **Target:** ADR Seriousness – Death (1) vs Non-Death (0)
- **Test Accuracy:** 0.902
- **Precision (DE Class):** 13.3%  
- **Recall (DE Class):** 5.9%

---

## 🧪 Key Results

### 📉 Deviation Tracking
19,967 records flagged with invalid therapy timelines (THER dataset).

We found that over 19,000 treatment records in the current dataset are using a placeholder date (January 1, 1970) instead of the actual therapy start or end dates. This issue can compromise our ability to verify treatment durations, assess quality risks, or meet regulatory audit expectations. These records should be reviewed and corrected to restore data reliability

### 🚨 CAPA Flagged Pairs
Most frequent drug-reaction combinations (e.g., `XOLAIR` – Pneumonia, `METHOTREXATE` – Arthralgia).

The CAPA dashboard highlights repeated adverse events associated with specific drugs, like INFLECTRA and OXYCONTIN. These high-frequency signals suggest potential misuse, side effects, or administration issues that may need quality review or regulatory escalation. Monitoring these trends proactively supports compliance with health authority expectations and strengthens risk-based quality management.

### ⚠️ Mismatch Detection
7036759 indication vs. reaction mismatches (e.g., `Hypertension` vs. `Type 2 Diabetes Mellitus`).

We have identified over 7 million records where the intended use of a drug does not match the adverse reaction reported. Some mismatches may reflect real safety issues, like panic attacks or abnormal behavior when treating unrelated conditions. Others may indicate off-label use or poor data capture. These mismatches should be reviewed to assess potential compliance or safety concerns.

### 🧠 Predictive Modeling
Model trained with `RandomForestClassifier` achieved 90.2% accuracy in predicting serious ADRs.

The predictive model we trained performs well on common, non-serious reactions but currently misses most serious ones. While it offers strong support for trend monitoring and triage, it should not be used for final safety decisions without more advanced modeling and richer patient data.
---

## 📋 Model Audit Summary
```json
{
  "Total Records": 69612,
  "Records Used in Modeling": 69612,
  "Features Used": ["age", "sex", "wt"],
  "Target": "ADR Seriousness: Death (1) vs Non-Death (0)",
  "Model Used": "RandomForestClassifier",
  "Test Accuracy": 0.902
}
```
The ADR Seriousness Prediction Model achieved 90.2% test accuracy using age, sex, and weight data from 69,612 reports. Built using a Random Forest Classifier, it helps flag high-risk ADR cases for review. While accurate overall, its simplicity limits detection of rare but severe outcomes, and it should be complemented by expert oversight or enhanced features for real-world deployment.
---

## 💡 Conceptual Enhancement – AGI Viewpoint
- Integrate **causal inference** models for ADR causality assessment.
- Use **temporal transformers** for therapy sequence modeling.
- Add **semantic matching** with LLMs for drug-reaction similarity detection.
- Enable **federated learning** for confidential PV network collaboration.

---

## 📚 References
- [FDA FAERS](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers)
- WHO UMC Signal Detection: [https://who-umc.org](https://who-umc.org)
- Liu et al., 2020: "Deep Learning for Drug-Induced Toxicity Prediction"

---

> 🔗 For the full Python code and Streamlit dashboard, please see `/dashboard_app.py`.
