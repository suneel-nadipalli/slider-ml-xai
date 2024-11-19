from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import pandas as pd

# Step 1: Load and preprocess the Iris dataset
def load_iris_data():
    """
    Load and preprocess the Iris dataset.
    Returns:
        X (DataFrame): Features for training.
        y (Series): Target variable (species).
    """
    data = load_iris(as_frame=True)
    X = data.data  # Features: sepal length, sepal width, petal length, petal width
    y = data.target  # Target: species (0, 1, 2 corresponding to Setosa, Versicolor, Virginica)
    return X, y, data.target_names

# Step 2: Train Decision Tree Classifier
def train_decision_tree_classifier(X, y):
    """
    Train a Decision Tree Classifier.
    Args:
        X (DataFrame): Features for training.
        y (Series): Target variable.
    Returns:
        model (DecisionTreeClassifier): Trained Decision Tree Classifier model.
        X_test, y_test: Test data for evaluation.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, target_names=load_iris().target_names)
    print("Classification Report:\n", report)
    return model, X_test, y_test

# Step 3: Save Model and Test Data
def save_model_and_test_data(model, X_test):
    """
    Save the trained model and test data.
    Args:
        model (DecisionTreeClassifier): Trained Decision Tree Classifier model.
        X_test (DataFrame): Test features.
        y_test (Series): Test target variable.
        target_names (list): Target class names.
        model_file (str): File path for saving the model.
        data_file (str): File path for saving the test data.
    """
    # Save the model
    with open("iris_tree_model.pkl", "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved")

    # Save the test dataset
    test_data = X_test.copy()
    test_data.to_csv("iris_test_data.csv", index=False)
    print(f"Test data saved")