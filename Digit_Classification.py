# Importing Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data Preprocessing
x_train = x_train / 255.0  # Normalize to [0, 1] range
x_test = x_test / 255.0    # Normalize to [0, 1] range

# Visualizing the Dataset
plt.figure(figsize=(10, 5))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Building the Model
model = Sequential([
    Flatten(input_shape=(28, 28)),    # Flatten the 28x28 input images
    Dense(128, activation='relu'),   # First hidden layer
    Dense(64, activation='relu'),    # Second hidden layer
    Dense(10, activation='softmax')  # Output layer for 10 classes (digits 0-9)
])

# Compiling the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the Model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.2f}")

# Plotting Training and Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the Model
model.save("digit_classifier_model.h5")
print("Model saved as 'digit_classifier_model.h5'.")

# Making Predictions
predictions = model.predict(x_test)

# Visualizing Predictions
plt.figure(figsize=(10, 5))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {tf.argmax(predictions[i]).numpy()}")
    plt.axis('off')
plt.tight_layout()
plt.show()
