import os
import pickle
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Convert labels to integers
unique_labels = np.unique(labels)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
labels = np.array([label_to_index[label] for label in labels])

print(f"Unique labels: {unique_labels}")
print(f"Number of classes: {len(unique_labels)}")

# Count samples per class
class_counts = Counter(labels)
print("Class counts before filtering:", class_counts)

# Remove classes with fewer than 2 samples
valid_classes = [cls for cls, count in class_counts.items() if count > 1]
indices = [i for i, label in enumerate(labels) if label in valid_classes]

data = data[indices]
labels = labels[indices]

print(f"Data shape after filtering: {data.shape}")
print(f"Labels shape after filtering: {labels.shape}")

# Update number of classes after filtering
num_classes = max(labels) + 1  # Fix: Calculate num_classes as max(labels) + 1
print(f"Number of classes after filtering: {num_classes}")

# One-hot encode the labels
labels = to_categorical(labels, num_classes=num_classes)

# Reshape data for LSTM
timesteps = 1
data = data.reshape((data.shape[0], timesteps, data.shape[1]))

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=np.argmax(labels, axis=1)
)
print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

# Define the LSTM model
model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(timesteps, data.shape[2])),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Evaluate the model
y_pred = model.predict(x_test)  # Ensure x_test is already reshaped correctly
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices
y_true = np.argmax(y_test, axis=1)  # Convert one-hot labels to class indices

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred_classes)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# Predict a single sample
single_sample = np.random.random((data.shape[2],))  # Example: A single sample with the correct feature size
single_sample = single_sample.reshape(1, timesteps, data.shape[2])  # Reshape to (1, 1, 42)
prediction = model.predict(single_sample)
predicted_class = np.argmax(prediction)
print(f"Predicted class: {predicted_class}")
