import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# Setting seeds for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

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

# Define a simple sequential model
def create_model(num_input_features, num_classes):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(num_input_features,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Determine the number of input features and classes
num_input_features = x_train.shape[1]
num_classes = len(pd.unique(y_train))

# Create a basic model instance
model = create_model(num_input_features, num_classes)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Predict the test set results
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Calculate the classification error
total = cm.sum()
correct = np.diag(cm).sum()
error_rate = (total - correct) / total * 100
print(f"Classification Error: {error_rate:.2f}%")
