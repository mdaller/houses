
import pandas as pd
train =pd.read_csv('data/train.csv')
#print(train.head())

#print(train.columns.tolist())
print("pool qc data:")
print(train['PoolQC'].isnull().value_counts())
print(train.shape)