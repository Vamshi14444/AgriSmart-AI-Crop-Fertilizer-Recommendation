# AgriSmart AI — Sustainable Crop & Fertilizer Advisor

> **Smarter Crop Choices. Balanced Nutrients. Sustainable Farming.**

---

## Abstract

AgriSmart AI is an end-to-end machine learning web application that helps farmers and agricultural students identify suitable crops and fertilizers based on measured soil nutrient levels, soil characteristics, and environmental conditions. The system trains, evaluates, and deploys real supervised classification models using publicly available agricultural datasets, and presents its recommendations through a polished, interactive Streamlit interface.

This project was developed as an academic machine learning prototype to demonstrate:
- Real supervised multiclass classification (not hardcoded predictions)
- Transparent model evaluation on held-out test data
- Explainable recommendations with feature importance
- Responsible handling of agricultural guidance
- Relevance to Sustainable Development Goals (SDGs 2, 12, 15)

---

## Problem Statement

Smallholder farmers in many regions make crop selection and fertilizer decisions based on tradition or local knowledge, often without access to soil testing or certified agronomic advice. This can lead to mismatched crop selections, excessive or insufficient fertilizer application, soil degradation, and economic losses.

A data-driven advisory tool can supplement — though never replace — local expert knowledge by identifying statistical patterns between soil characteristics and successful crop outcomes.

---

## Objectives

1. Build a supervised ML system for crop recommendation using soil and climate measurements.
2. Build a separate supervised ML system for fertilizer category prediction.
3. Train, compare, and evaluate multiple classification algorithms with honest metrics.
4. Present explainable predictions with feature importance and confidence scores.
5. Provide general sustainability guidance aligned with SDG 2, 12, and 15.
6. Deliver a usable, well-designed web interface for farmers and students.

---

## Existing System Limitations

- Static lookup tables or hardcoded recommendations lack generalisation.
- Rule-based systems require expert knowledge for every crop-soil combination.
- Most public tools do not display evaluation metrics or explain their reasoning.
- Many academic projects claim trained models but use hardcoded outputs.

---

## Proposed System & New Capabilities

AgriSmart AI addresses these limitations through an integrated precision agtech platform:
- **Trained ML Models**: 3 algorithms evaluated with zero data leakage (Random Forest, Decision Trees, KNN).
- **Explainability**: Gini feature attribution bar charts for transparency.
- **🌡️ Climate "What-If" Sensitivity Simulator**: Real-time stress testing against droughts (-50% rainfall) and heatwaves (+4°C), featuring dynamic multi-crop probability transition curves.
- **⚖️ Head-to-Head Crop Comparison**: Side-by-side comparative matrix across 15 agronomic dimensions plus visual dual-radar diagrams.
- **📅 Seasonal Crop Calendar**: Interactive 12-month phenology schedule covering Kharif (Monsoon), Rabi (Winter), and Zaid (Summer) sowing/harvesting windows.
- **📄 Downloadable PDF Field Advisory**: Branded, timestamped PDF reports generated on-demand with ReportLab for farm audits or printouts.
- **Targeted Fertilization**: Independent pipeline predicting formulation grades without fabricated dosages.
- **UN SDG Alignment**: Transparently supports SDG 2 (Zero Hunger), SDG 12 (Responsible Consumption), and SDG 15 (Life on Land).

---

## System Architecture

```
User Input (Streamlit UI)
         │
         ▼
  Input Validation
         │
         ▼
  Preprocessing Pipeline (fitted on training data only)
         │
         ├── Numeric features → Passthrough or StandardScaler
         └── Categorical features → OrdinalEncoder (handles unseen categories)
         │
         ▼
  Trained Classifier (Decision Tree / Random Forest / KNN)
         │
         ▼
  Prediction + Confidence + Feature Importance
         │
         ▼
  Recommendation Display + Sustainability Guidance + History
```

---

## Dataset Descriptions and Sources

### Crop Recommendation Dataset
- **Source**: Kaggle — [Crop Recommendation Dataset by Atharva Ingle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Rows**: ~2,200
- **Target**: crop label (22 classes)
- **Features**: N, P, K, temperature, humidity, ph, rainfall

### Fertilizer Prediction Dataset
- **Source**: Kaggle — [Fertilizer Prediction](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction)
- **Rows**: ~99
- **Target**: Fertilizer Name (7 classes)
- **Features**: Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous

> ⚠️ These datasets are provided by the Kaggle community. Always verify dataset provenance and quality before use in production systems.

---

## Feature Descriptions

### Crop Model Features

| Feature | Description | Unit |
|---|---|---|
| N | Soil nitrogen content | kg/ha |
| P | Soil phosphorus content | kg/ha |
| K | Soil potassium content | kg/ha |
| temperature | Average ambient temperature | °C |
| humidity | Relative humidity | % |
| ph | Soil pH | 0–14 scale |
| rainfall | Annual/seasonal rainfall | mm |

### Fertilizer Model Features

| Feature | Description |
|---|---|
| Temperature | Ambient temperature (°C) |
| Humidity | Relative humidity (%) |
| Moisture | Soil moisture (%) |
| Soil Type | Categorical: Sandy, Loamy, Black, Red, Clayey |
| Crop Type | Categorical: Maize, Sugarcane, Cotton, etc. |
| N / K / P | Soil nitrogen, potassium, phosphorus content |

---

## Machine Learning Algorithms

### Algorithms Trained and Compared

| Algorithm | Notes |
|---|---|
| Decision Tree | Interpretable; feature importance available; no scaling needed |
| Random Forest | Ensemble of trees; robust; feature importance available |
| K-Nearest Neighbours | Instance-based; requires feature scaling |

### Model Selection
The best model is selected by **weighted F1-score on the held-out test set** (20% of data, stratified).

---

## Preprocessing Methodology

1. **Column normalisation**: Column names are canonicalised (lowercase, trimmed, spelling variants resolved — e.g. `Temparature` → `temperature`).
2. **Duplicate detection**: Exact duplicate rows are reported and removed only after inspection.
3. **Missing value handling**: Rows with non-numeric values in numeric feature columns are coerced to NaN and dropped with a warning.
4. **Categorical encoding**: OrdinalEncoder with `handle_unknown='use_encoded_value'` (unseen categories return -1 at inference, not an error).
5. **Scaling**: StandardScaler applied only for KNN pipelines; tree-based models use raw features.
6. **Pipeline**: All preprocessing is fitted only on training data and applied to test data, preventing data leakage.

---

## Model Evaluation Methodology

- **Split**: Stratified 80/20 train/test split using a fixed random seed (42) for reproducibility.
- **Metrics**: Accuracy, Precision (weighted), Recall (weighted), F1-Score (weighted).
- **Confusion matrix**: Computed on the test set.
- **Comparison**: All three algorithms are evaluated; the best is saved.
- **No training accuracy is reported** — only test set results, to prevent misleading claims.

---

## Installation Instructions (Windows)

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Git (optional)
- VS Code (recommended)

### Steps

```powershell
# 1. Navigate to the project directory
cd "C:\Users\YourName\path\to\crop-fertilizer-ml"

# 2. Create a virtual environment (if not already done)
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt
```

> **Note for Python 3.13+**: All dependencies in `requirements.txt` are compatible with Python 3.13 and 3.14. If you encounter issues with a specific package, check its PyPI page for the latest compatible version.

---

## Exact Commands to Run the Project

### Activate virtual environment (PowerShell)
```powershell
.venv\Scripts\Activate.ps1
```

### Inspect downloaded datasets
```powershell
python scripts/inspect_data.py
```

### Train the crop model
```powershell
python scripts/train_crop_model.py
```

### Train the fertilizer model
```powershell
python scripts/train_fertilizer_model.py
```

### Run automated tests
```powershell
python -m pytest tests/ -v
```

### Start the Streamlit web application
```powershell
streamlit run app.py
```

---

## How to Add Downloaded Datasets

1. Download the CSV or ZIP files from the Kaggle links listed above.
2. Place them in the `datasets/` folder:
   ```
   crop-fertilizer-ml/
   └── datasets/
       ├── Crop_recommendation.csv
       └── Fertilizer Prediction.csv
   ```
3. The application automatically discovers all CSV files and extracts ZIP archives.
4. Run `python scripts/inspect_data.py` to verify the datasets are recognised correctly.

---

## How to Retrain the Models

After updating or replacing datasets:
```powershell
python scripts/train_crop_model.py
python scripts/train_fertilizer_model.py
```

Models are saved to `models/`. Restart the Streamlit app to load the new models.

---

## Project Structure

```
crop-fertilizer-ml/
│
├── app.py                      # Streamlit main application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore
│
├── datasets/
│   └── README.md               # Dataset download instructions
│
├── models/                     # Saved model files (not in git)
│   ├── crop_model.joblib
│   ├── crop_metadata.json
│   ├── fertilizer_model.joblib
│   └── fertilizer_metadata.json
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset discovery, ZIP extraction, schema detection
│   ├── preprocessing.py        # Feature engineering, pipeline construction
│   ├── crop_model.py           # Crop model training, saving, loading, prediction
│   ├── fertilizer_model.py     # Fertilizer model training, saving, loading, prediction
│   ├── evaluation.py           # Classification metrics and comparison
│   ├── recommendations.py      # Crop/fertilizer explanation generation
│   ├── report_generator.py     # PDF Field Advisory generator (ReportLab)
│   ├── crop_comparison.py      # Head-to-Head agronomic matrix
│   ├── crop_calendar.py        # Seasonal timeline Gantt chart (Kharif/Rabi/Zaid)
│   ├── sustainability.py       # Sustainability tips, checklist, SDG relevance
│   └── history.py              # Prediction history (local CSV storage)
│
├── scripts/
│   ├── inspect_data.py         # Dataset inspection utility
│   ├── train_crop_model.py     # Crop model training script
│   └── train_fertilizer_model.py # Fertilizer model training script
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_crop_model.py
│   ├── test_fertilizer_model.py
│   └── test_new_features.py
│
└── artifacts/
    └── prediction_history.csv  # Stored predictions (not in git)
```

---

## Screenshots

> 📸 Add screenshots after running the application.

| Page | Description |
|---|---|
| Home Dashboard | System status, quick-start guide, SDG alignment |
| Crop Recommendation | Input form, prediction result, top-3 crops, feature importance |
| Fertilizer Recommendation | Fertilizer prediction, sustainability guidance |
| Soil Health Analysis | Nutrient gauges, NPK radar chart |
| Sustainability Advisor | Tips, interactive checklist, SDG descriptions |
| Model Performance | Metrics table, algorithm comparison chart |
| Prediction History | Downloadable prediction log |

---

## Limitations

- Trained on public datasets that may not represent all regions or soil types.
- No fertilizer dosage quantities are provided — only categorical predictions.
- The system does not use live weather, satellite, or regional soil data.
- Predictions are statistical patterns from training data, not agronomic guarantees.
- Small fertilizer datasets may limit generalisation across crop types.
- Models do not account for pest pressure, irrigation availability, or market factors.

---

## Future Enhancements

- Integration with regional soil health databases (e.g. India Soil Health Card data)
- Live weather API integration for temperature and humidity
- Crop-specific nutrient requirement tables for dose recommendations
- Mobile-responsive Progressive Web App deployment
- Multi-language interface for local farmer accessibility
- Satellite NDVI integration for field health monitoring

---

## Sustainable Development Goals Relevance

| SDG | Relevance |
|---|---|
| **SDG 2 — Zero Hunger** | Data-driven crop selection to support more productive and reliable food production. |
| **SDG 12 — Responsible Consumption and Production** | Encourages targeted fertilizer use, reducing overuse, environmental waste, and input costs. |
| **SDG 15 — Life on Land** | Promotes soil health, crop rotation, and practices that reduce land degradation. |

> ⚠️ **Disclaimer**: The project aligns with the stated SDGs in its educational goals. No claims are made about measured or quantified real-world impact, as this is an academic prototype.

---

## Academic Disclaimer

This application is an educational prototype submitted in partial fulfilment of academic requirements. It is not a substitute for certified agricultural advice, laboratory soil testing, or professional agronomic consultation. All crop and fertilizer recommendations produced by the machine learning models should be verified with appropriate local authorities, agricultural extension services, and certified soil testing laboratories before any farming decision is made.

No fertilizer dosage quantities or yield guarantees are provided by this system.

---

*AgriSmart AI — Academic ML Project · Python · Scikit-learn · Streamlit*
