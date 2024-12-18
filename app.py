import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

# Function to load stock data
def load_stock_data(file_path):
    """Loads stock data from an Excel file."""
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Function to preprocess input data
def preprocess_data(data, target_column):
    """
    Preprocess the stock data.
    :param data: DataFrame containing stock data
    :param target_column: Column to predict
    :return: X (features), y (target)
    """
    try:
        y = data[target_column]
        X = data.drop(columns=[target_column])
        return X, y
    except KeyError as e:
        print(f"Target column {target_column} not found: {e}")
        return None, None

# Define the pipeline
def create_pipeline():
    """Creates a Scikit-learn pipeline with preprocessing and a regression model."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize the features
        ('regressor', LinearRegression())  # Linear Regression model
    ])
    return pipeline

# Main function to process stock data and predict
def process_stock(file_path, target_column):
    """Processes stock data and makes predictions."""
    data = load_stock_data(file_path)
    if data is None:
        return f"Error: Could not load file {file_path}"

    X, y = preprocess_data(data, target_column)
    if X is None or y is None:
        return f"Error: Invalid data in {file_path}"

    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create and fit the pipeline
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    # Ensure the pipeline is fitted
    try:
        check_is_fitted(pipeline)
    except Exception as e:
        return f"Error: Pipeline not fitted for {file_path}. Details: {e}"

    # Make predictions
    try:
        predictions = pipeline.predict(X_test)
        print(f"Predictions for {file_path}: {predictions}")
        return predictions
    except Exception as e:
        return f"Error predicting for {file_path}. Details: {e}"

# Process multiple stock files
def process_multiple_stocks(file_paths, target_column):
    """Processes multiple stock files."""
    for file_path in file_paths:
        result = process_stock(file_path, target_column)
        print(f"{file_path}: {result}")

# Example usage
if __name__ == "__main__":
    # List of stock files
    stock_files = [
        "AJANTPHARM.xlsx",
        "AUBANK.xlsx"
    ]
    # Target column to predict
    target_column = "Close_Price"  # Replace with your actual target column name

    # Process all stocks
    process_multiple_stocks(stock_files, target_column)
