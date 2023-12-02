import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Setting seeds for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load your dataset
train_data = pd.read_csv('A2-ring-merged.csv')
test_data = pd.read_csv('A2-ring-test.csv')

# Assuming the last column is the target variable
x_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
x_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Convert categorical labels to numbers if necessary
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Standardize the dataset
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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

# Calculate ROC Curve and AUC for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])  # Change here for binary classification
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
