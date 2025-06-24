
import pandas as pd
train =pd.read_csv('data/train.csv')
#print(train.head())

#print(train.columns.tolist())

#print(f"{column} data:")
print(train.isna().sum().sort_values(ascending=False))
print("pool unique")
print(train['PoolArea'].unique())
print((train['PoolArea'] == 0).sum())
#pool area has 0 for no pool and matches the PoolQC NaN values
#so we can drop PoolQC
train.drop(['PoolQC'])