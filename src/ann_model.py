import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Load Training Dataset
# -----------------------------
train_data = pd.read_excel(
    "Afreen_Training_Data.xlsx"
)

X_train = train_data[
    ['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4']
].values

y_train_labels = train_data['Target label'].values

# -----------------------------
# Load Testing Dataset
# -----------------------------
test_data = pd.read_excel(
    "Afreen_Testing_Data.xlsx"
)

test_data = test_data.dropna(
    subset=['Target label']
)

X_test = test_data[
    ['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4']
].values

y_test_labels = test_data['Target label'].values

# -----------------------------
# Encode Target Labels
# -----------------------------
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(
    y_train_labels
)

y_test_encoded = label_encoder.transform(
    y_test_labels
)

y_train = to_categorical(y_train_encoded)
y_test = to_categorical(y_test_encoded)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Build ANN Model
# -----------------------------
model = Sequential([

    Dense(
        8,
        activation='relu',
        input_dim=4
    ),

    Dense(
        4,
        activation='relu'
    ),

    Dense(
        y_train.shape[1],
        activation='softmax'
    )
])

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=8,
    verbose=1
)

# -----------------------------
# Evaluate Model
# -----------------------------
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Accuracy: "
      f"{accuracy * 100:.2f}%")

# -----------------------------
# Generate Predictions
# -----------------------------
y_pred = model.predict(X_test)

pred_classes = np.argmax(y_pred, axis=1)
true_classes = np.argmax(y_test, axis=1)

pred_labels = label_encoder.inverse_transform(
    pred_classes
)

true_labels = label_encoder.inverse_transform(
    true_classes
)

# -----------------------------
# Save Testing Results
# -----------------------------
results = pd.DataFrame({

    'True Label': true_labels,

    'Predicted Label': pred_labels,

    'Match':
        pred_labels == true_labels
})

results.to_excel(
    "ANN_Test_Results.xlsx",
    index=False
)

print("\nTesting results saved successfully.")
