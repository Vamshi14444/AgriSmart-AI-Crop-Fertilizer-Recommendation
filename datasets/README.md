# Datasets Directory

Place your downloaded datasets here. The application supports CSV files and ZIP archives.

## Recommended Public Datasets

### Crop Recommendation Dataset
- **Source**: [Kaggle – Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **File**: `Crop_recommendation.csv`
- **Columns**: N, P, K, temperature, humidity, ph, rainfall, label

### Fertilizer Prediction Dataset
- **Source**: [Kaggle – Fertilizer Prediction](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction)
- **File**: `Fertilizer Prediction.csv`
- **Columns**: Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous, Fertilizer Name

## How to Add Datasets

1. Download the CSV or ZIP files from the sources above.
2. Place them in this `datasets/` folder.
3. If you have a ZIP file, the application will extract it automatically.
4. Run the application: `streamlit run app.py`

## Notes

- The application inspects actual column names — it does not assume dataset content by filename.
- Multiple CSV files can coexist; the application will identify them by their columns.
- Do not rename columns manually; the application handles common variations.
