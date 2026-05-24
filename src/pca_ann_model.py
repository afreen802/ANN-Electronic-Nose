
# PCA + ANN PROGRAM

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# STEP 1: Load PCA TRAINING dataset
train_data = pd.read_excel("PCA_Training_Output.xlsx")

X_train = train_data[['PC1', 'PC2', 'PC3']].values
y_train_labels = train_data['Target label'].values

# STEP 2: Load PCA TESTING dataset
test_data = pd.read_excel("PCA_Testing_Output.xlsx")

# Remove rows with missing labels (safety)
test_data = test_data.dropna(subset=['Target label'])

X_test = test_data[['PC1', 'PC2', 'PC3']].values
y_test_labels = test_data['Target label'].values

# STEP 3: Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_labels)
y_test_encoded = label_encoder.transform(y_test_labels)

y_train = to_categorical(y_train_encoded)
y_test = to_categorical(y_test_encoded)

num_classes = y_train.shape[1]

# STEP 4: Scale PCA features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 5: Build ANN
model = Sequential([
    Dense(8, activation='relu', input_dim=3),
    Dense(6, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# STEP 6: Train ANN (PCA TRAINING data)
print("\n⏳ Training PCA+ANN...")
model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=8,
    verbose=1
)

# STEP 7: Test ANN (PCA TESTING data)
print("\n🧪 Testing PCA+ANN...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# STEP 8: Predictions on testing data
y_pred = model.predict(X_test)

pred_classes = np.argmax(y_pred, axis=1)
true_classes = np.argmax(y_test, axis=1)

pred_labels = label_encoder.inverse_transform(pred_classes)
true_labels = label_encoder.inverse_transform(true_classes)

# STEP 9: Save PCA+ANN testing results
results = test_data.copy()
results['Out1'] = y_pred[:, 0]
results['Out2'] = y_pred[:, 1]
results['Out3'] = y_pred[:, 2]
results['Pred_Label'] = pred_labels
results['Match'] = pred_labels == true_labels

output_path = "/Applications/Data/Final_PCA_ANN_Testing_Output.xlsx"
results.to_excel(output_path, index=False)

print(" PCA+ANN testing output saved at:")
print(output_path)
