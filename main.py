
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


train =pd.read_csv('data/train.csv')
print(train.head())
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
#print(train.columns.tolist())
correlation=train[numerical_cols].corr()['SalePrice']
print("correlation scores")
print(correlation.sort_values(ascending=False).head(80))
#print("correlation scores" +correlation)
#print(f"{column} data:")
