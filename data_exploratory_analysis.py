print(train.isna().sum().sort_values(ascending=False))
print("missing values")
print(train.isnull().sum().sort_values(ascending=False))
print("pool unique")
print("fence to see missing values :"   )
#print(train['Fence'].head(20))
#print(train['PoolArea'].unique())
#print((train['PoolArea'] == 0).sum())
#pool area has 0 for no pool and matches the PoolQC NaN values
#so we can drop PoolQC
train.drop(['PoolQC'])
#print(train['BldgType'].value_counts().sort_values(ascending=False))
#drop Fence as it has too many missing values
train.drop(['Fence'], axis=1, inplace=True)