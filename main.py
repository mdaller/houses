
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


train =pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print(train.head())
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
#print(train.columns.tolist())
correlation=train[numerical_cols].corr()['SalePrice']
print("correlation scores")
print(correlation.sort_values(ascending=False).head(80))
#print("correlation scores" +correlation)
#print(f"{column} data:")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load your data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Save test IDs for submission
test_ids = test['Id']

# Separate target variable and drop it from train features
y = np.log1p(train['SalePrice'])  # log-transform target
X = train.drop(columns=['SalePrice', 'Id'])  # drop Id too
X_test = test.drop(columns=['Id'])

# Combine train and test for consistent preprocessing
combined = pd.concat([X, X_test], keys=["train", "test"])

# Handle missing values (simple strategy here — you can customize per column if needed)
# Separate numerical and categorical columns
num_cols = combined.select_dtypes(include=['number']).columns
cat_cols = combined.select_dtypes(include=['object']).columns

# Impute missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

combined[num_cols] = num_imputer.fit_transform(combined[num_cols])
combined[cat_cols] = cat_imputer.fit_transform(combined[cat_cols])

# One-hot encode categorical variables
combined = pd.get_dummies(combined)

# Split combined back to train/test
X = combined.loc["train"]
X_test = combined.loc["test"]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on validation
y_val_pred = rf.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("Validation RMSE:", val_rmse)

# Predict on test
y_test_pred = rf.predict(X_test)
y_test_pred = np.expm1(y_test_pred)  # Inverse log1p

# Save to submission.csv
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': y_test_pred})
submission.to_csv('submission.csv', index=False)
# Display the first few rows of the submission file
print(submission.head())