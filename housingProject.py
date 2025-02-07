import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Lagos Housing dataset
data = pd.read_csv("LagosHousingData.csv")

# Print the first 5 rows of the dataset
print(data.head())

# Convert categorical variables to dummy variables
data = pd.get_dummies(data)

# Handle missing values by filling them with the mean of the column
data = data.fillna(data.mean())

# Drop any remaining rows with NaNs (just in case)
#data = data.dropna()

# Check for remaining NaN values
#print(data.isnull().sum())

# Separate the features (X) and the target variable (y)
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")





# Input X contains NaN indicates that there are 
# missing values (NaNs) in your dataset. 
# You need to handle these missing values before fitting the model. 
# You can either remove rows with missing values or fill them with a specific value 
# (e.g., the mean of the column).
