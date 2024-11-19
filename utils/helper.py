import pickle, shap
import pandas as pd

import sys

sys.path.append("..")

def load_model(file_path):
    try:
        return pickle.load(file_path)
    except TypeError:
        file = open(file_path, "rb")
        return pickle.load(file)

def load_dataset(file_path):
    return pd.read_csv(file_path)

def predict(model, data):
    return model.predict(data)

def update_prediction(model, row, slider_values):
    print(f"Slider values: {slider_values}")
    updated_row = row.copy()
    for col, val in slider_values.items():
        updated_row[col] = val
    return model.predict([updated_row])

def get_coefficients(model, feature_names):
    if hasattr(model, 'coef_'):
        coefficients = model.coef_
        return dict(zip(feature_names, coefficients))
    elif hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
        return dict(zip(feature_names, feature_importance))
    else:
        raise ValueError("Model does not have coefficients. Try SHAP for interpretability.")

def compute_shap_values(model, X_sample):
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    return shap_values
