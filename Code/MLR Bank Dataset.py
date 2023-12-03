import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# Setting seeds for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)

# Load your dataset
train_data = pd.read_csv('bank-additional-training.csv')
test_data = pd.read_csv('bank-additional-test.csv')

# Separate the numeric columns for scaling
numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# Extract the numeric columns from the training and testing datasets
x_train_numeric = train_data[numeric_columns].values
x_test_numeric = test_data[numeric_columns].values

# Encode categorical features
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')  # Add handle_unknown='ignore'

# Fit and transform the encoder on the training data
x_train_categorical = encoder.fit_transform(train_data[categorical_columns])
x_test_categorical = encoder.transform(test_data[categorical_columns])

# Combine numeric and encoded categorical features
x_train = np.hstack((x_train_numeric, x_train_categorical))
x_test = np.hstack((x_test_numeric, x_test_categorical))

# Assuming the last column is the target variable
y_train = train_data.iloc[:, -1].values
y_test = test_data.iloc[:, -1].values

# Preprocess the target variable to ensure it contains only 1s and 0s
y_train = np.where(y_train == 'yes', 1, 0)
y_test = np.where(y_test == 'yes', 1, 0)

# Standardize only the numeric columns
scaler = StandardScaler()
x_train_numeric = scaler.fit_transform(x_train_numeric)
x_test_numeric = scaler.transform(x_test_numeric)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(x_train, y_train)

# Predict the target variable for the test data
y_pred = model.predict(x_test)

# Convert predicted values to binary (0 or 1) based on a threshold (e.g., 0.5)
y_pred_classes = (y_pred >= 0.5).astype(int)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Calculate the classification error
total = cm.sum()
correct = np.diag(cm).sum()
error_rate = (total - correct) / total * 100
print(f"Classification Error: {error_rate:.2f}%")
