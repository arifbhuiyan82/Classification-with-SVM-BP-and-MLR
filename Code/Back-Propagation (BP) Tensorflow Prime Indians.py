import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np

# Load your dataset
train_data = pd.read_csv('pima-indians-diabetes-training.csv')
test_data = pd.read_csv('pima-indians-diabetes-test.csv')

# Separate features and target variable
x_train = train_data.iloc[:, :-1].values
x_test = test_data.iloc[:, :-1].values

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Target variable
y_train = train_data.iloc[:, -1].values
y_test = test_data.iloc[:, -1].values

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
num_classes = 2  # Since it's binary classification

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
