# House Price Prediction

This project trains a house price regression model using a scikit-learn pipeline and then runs inference on new data.

## What this repo contains

- `main.py` - main script for training and inference
- `housing.csv` - dataset used for training (place in the same folder as `main.py`)
- `test.csv` - optional input data for inference
- `test_output.csv` - generated output predictions after inference
- `.gitignore` - ignores large model artifacts like `model.pkl` and `pipeline.pkl`

## How it works

### Training mode
The script enters training mode when `model.pkl` does not exist.
It will:

1. Load `housing.csv`
2. Create stratified train/test splits based on `median_house_value`
3. Prepare features with a pipeline:
   - median imputation for numeric values
   - standard scaling for numeric values
   - one-hot encoding for `ocean_proximity`
4. Train a `RandomForestRegressor`
5. Save `model.pkl` and `pipeline.pkl`

### Inference mode
When `model.pkl` already exists, the script will:

1. Load `model.pkl` and `pipeline.pkl`
2. Read `test.csv`
3. Transform the test data using the saved pipeline
4. Predict `median_house_value`
5. Save predictions to `test_output.csv`

## How to run

1. Place `housing.csv` and `test.csv` in the same folder as `main.py`
2. Run the script:

```powershell
python main.py
```

3. If `model.pkl` does not exist, the script will train and save the model.
4. If `model.pkl` exists, the script will perform inference and create `test_output.csv`.

## Notes

- `model.pkl` is a large binary model file and should not be committed to the repo.
- `pipeline.pkl` is also ignored by `.gitignore` after cleaning the repo.
- Keep the dataset files and script together in the same folder when running.

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- joblib

Install dependencies with:

```powershell
pip install pandas numpy scikit-learn joblib
```
