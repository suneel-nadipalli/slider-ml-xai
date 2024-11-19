from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Step 1: Load the dataset
def load_data():
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target  # House prices
    return X, y

# Step 2: Train a regression model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return model, X_test, y_test

# Step 3: Save the trained model
def save_model(model, file_name="house_price_model.pkl"):
    import pickle
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {file_name}")


