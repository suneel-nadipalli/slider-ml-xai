# Slider-ML

Slider-ML is an interactive Streamlit-based web application designed to explore and interpret machine learning models in real-time. It allows users to adjust feature values dynamically and visualize the effects on predictions and class probabilities. This tool bridges the gap between model training and interpretability, making machine learning insights more accessible and actionable.

---

## Features / How to Use

### Key Features
- **Regression and Classification Support**:
  - Dedicated tabs for analyzing regression and classification models.
- **Dynamic Interaction**:
  - Adjust feature values via sliders and observe real-time changes in predictions or class probabilities.
- **Explainability Tools**:
  - **Regression**: Coefficients for linear models and feature importance for tree-based models.
  - **Classification**: Visualized class probabilities and feature contributions.
- **Preloaded Models and Datasets**:
  - Quick-start with built-in datasets and pre-trained models.
- **Custom Model Uploads**:
  - Use your own `.pkl` models and `.csv` datasets.

### Steps to Use
1. Navigate to the **Regression** or **Classification** tab.
2. Choose between:
   - **Preloaded Models and Datasets** for quick testing.
   - **Upload Files** to upload your own `.pkl` model and `.csv` dataset.
3. Click `Load New Sample` to analyze a random data point.
4. Use sliders to adjust feature values and view updated predictions or class probabilities in real-time.
5. Explore insights through visualized feature contributions or coefficients.

---

## How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/suneel-nadipalli/slider-ml-xai.git
cd slider-ml-xai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

### 4. Open in Your Browser
Streamlit will display a URL (e.g., http://localhost:8501). Open this link in your browser to use the application.

## How It Works

### Architecture
- Streamlit Framework:
    - Provides an intuitive and interactive user interface.
    - Facilitates seamless integration of sliders, visualizations, and model predictions.
- Regression and Classification Modules:
    - Regression: Supports Linear Regression and Decision Tree Regression.
    - Classification: Supports Logistic Regression and Decision Tree Classifiers.
- Real-Time Interaction:
    - Dynamically adjusts predictions or probabilities when feature sliders are moved.
    - Ensures visual and numerical updates occur instantaneously for an engaging user experience.

### How Coefficients Are Calculated

Coefficients are numerical values that represent the relationship between a feature and the target variable in linear models. 

1. **Linear Regression Coefficients**:

- When a linear regression model is uploaded or preloaded, its ```coef_``` attribute is accessed:
    ```bash
    coefficients = model.coef_
    ```
- Each coefficient corresponds to a feature in the dataset. For example, if the model predicts house prices and has features like rooms and area, the coefficients indicate how much a unit change in each feature affects the predicted price.
- These coefficients are displayed alongside each feature in the app to help users understand the feature's impact.

2. **Tree-Based Feature Importance:**

- For decision tree models, coefficients are not available because these models use splits rather than weights. Instead, feature importance is calculated using the ```feature_importances_``` attribute:
```bash
feature_importances = model.feature_importances_
```
- This value quantifies how much a feature contributes to reducing the impurity (e.g., variance or Gini index) across all splits in the tree.

3. **Classification Probabilities:**

- The Classification models use ```predict_proba``` to generate the probabilities for each class.
- And ```model.classes_``` provides the names of the classes
- The split-bar adjusts dynamically based on user input, providing a clear visual representation of how likely the sample belongs to each class.

## Folder Sturcture

Here's the folder structure for the repository:
```
slider-ml-xai/
├── app.py                 # Main application
├── utils/
│   ├── helper.py          # Functions for model and dataset loading
│   ├── ui/
│       ├── reg_components.py  # Regression UI components
│       ├── clf_components.py  # Classification UI components
│       ├── reg_screen.py  # Main Regression UI
│       ├── clf_screen.py  # Main Classification UI
│   ├── train/             # Scripts to train example models
│       ├── train_dec_tree_reg.py
│       ├── train_dec_tree_clf.py
│       ├── train_log_reg.py
│       ├── train_reg.py
├── data/                  # Preloaded datasets
├── models/                # Preloaded models
├── requirements.txt       # Python dependencies
```

## Preloaded Models and Datasets

| Model        | Dataset           | Type  |
|:-------------:|:-------------:|:-----:|
| Linear Regressor | [Boston Housing Dataset](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset) | Regression |
| Decision Tree Regressor | [Auto MPG Dataset](https://www.kaggle.com/datasets/uciml/autompg-dataset) | Regression |
| Logistic Regressor | [Titanic Dataset](https://www.kaggle.com/c/titanic/data) | Classification |

## Website link

Play around with the tool: https://slider-ml-xai.streamlit.app/