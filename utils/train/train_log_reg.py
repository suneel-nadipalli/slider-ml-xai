import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Step 1: Load and preprocess the Titanic dataset using seaborn
def load_and_preprocess_titanic_data():
    """
    Load and preprocess the Titanic dataset using seaborn.
    Returns:
        X (DataFrame): Features for training.
        y (Series): Target variable (Survived).
    """
    # Load dataset
    data = sns.load_dataset('titanic')
    
    # Select relevant features
    features = ["pclass", "sex", "age", "fare", "embarked"]
    target = "survived"
    
    # Drop rows with missing target
    data = data.dropna(subset=[target])
    
    # Handle missing values
    data["age"].fillna(data["age"].median(), inplace=True)
    data["fare"].fillna(data["fare"].median(), inplace=True)
    data["embarked"].fillna(data["embarked"].mode()[0], inplace=True)
    
    # Convert categorical features to numeric
    data["sex"] = data["sex"].map({"male": 0, "female": 1})
    data["embarked"] = data["embarked"].map({"C": 0, "Q": 1, "S": 2})
    
    # Split features and target
    X = data[features]
    y = data[target]
    
    return X, y

# Step 2: Train Logistic Regression model
def train_logistic_regression(X, y):
    """
    Train a logistic regression model.
    Args:
        X (DataFrame): Features for training.
        y (Series): Target variable.
    Returns:
        model (LogisticRegression): Trained logistic regression model.
        X_test, y_test: Test data for evaluation.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    print("Classification Report:\n", report)
    
    return model, X_test, y_test

# Step 3: Save Logistic Regression model and test data
def save_logistic_model_and_data(model, X_test, y_test):
    """
    Save the trained model and test data.
    Args:
        model (LogisticRegression): Trained logistic regression model.
        X_test (DataFrame): Test features.
        y_test (Series): Test target variable.
    """
    # Save the model
    with open("logistic_titanic_model.pkl", "wb") as file:
        pickle.dump(model, file)
    print("Logistic Regression model saved as 'logistic_titanic_model.pkl'")
    
    # Save the test dataset
    test_data = X_test.copy()
    test_data["Survived"] = y_test
    test_data.to_csv("titanic_test_data.csv", index=False)
    print("Test data saved as 'titanic_test_data.csv'")