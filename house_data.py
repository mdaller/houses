import pandas as pd
import numpy as np
def load_data(train, test):
    """
    Load the training and test datasets from CSV files.
    
    Parameters:
    - test.csv: Path to the test dataset CSV file.
    - train.csv: Path to the training dataset CSV file.
    
    Returns:
    - X_train: Features of the training dataset.
    - y_train: Target variable of the training dataset.
    - X_test: Features of the test dataset.
    """
    train = pd.read_csv(train)
    test = pd.read_csv(test)
    
    X_train = train.drop(columns=['target'])
    y_train = train['target']
    
    X_test = test
    
    return X_train, y_train, X_test