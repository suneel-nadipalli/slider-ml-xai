import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import pandas as pd

# Step 1: Load and preprocess the Auto MPG dataset
def load_auto_mpg_data():
    """
    Load and preprocess the Auto MPG dataset.
    Returns:
        X (DataFrame): Features for training.
        y (Series): Target variable (mpg).
    """
    data = sns.load_dataset('mpg').dropna()
    X = data[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']]
    y = data['mpg']
    return X, y

# Step 2: Train Decision Tree Regressor
def train_decision_tree_regressor(X, y):
    """
    Train a Decision Tree Regressor.
    Args:
        X (DataFrame): Features for training.
        y (Series): Target variable.
    Returns:
        model (DecisionTreeRegressor): Trained Decision Tree Regressor model.
        X_test, y_test: Test data for evaluation.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    return model, X_test, y_test

# Step 3: Save the model and test data
def save_model_and_test_data(model, X_test, y_test, model_file="reg_tree_miles.pkl", data_file="reg_tree_mile.csv"):
    with open(model_file, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved as '{model_file}'")
    test_data = X_test.copy()
    test_data['mpg'] = y_test
    test_data.to_csv(data_file, index=False)
    print(f"Test data saved as '{data_file}'")
