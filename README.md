# MNIST Handwritten Digit Recognition

CNN-based digit classifier achieving 99% accuracy on MNIST dataset.

## Model Architecture

- 3 Conv2D layers (64 filters each, 3x3 kernel)
- 2 MaxPooling layers (2x2)
- Dropout (0.5) for regularization
- Dense layer with softmax (10 classes)

## Quick Start

bash
pip install tensorflow numpy matplotlib
python
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers, models

## Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images_processed = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
train_labels_categorical = to_categorical(train_labels)

# Build model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images_processed, train_labels_categorical, epochs=5, batch_size=64)
Making Predictions
python

# Predict single image
def predict_image(index):
    pred = model.predict(test_images_processed[index:index+1])
    digit = np.argmax(pred)
    confidence = np.max(pred) * 100
    print(f"Predicted: {digit} | Actual: {test_labels[index]} | Confidence: {confidence:.2f}%")
    return digit

# Predict first 10 test images
predictions = model.predict(test_images_processed[:10])
for i in range(10):
    print(f"Image {i+1}: Predicted = {np.argmax(predictions[i])}, Actual = {test_labels[i]}")
    
## Results
Test Accuracy: 99.2%
Training: 5 epochs, 64 batch size

Loss: 0.0254

# Key Functions
Function	            Purpose
- predict_image(index)	 - Predict digit by test image index
- model.predict()	        - Batch predictions
- np.argmax()	           - Get predicted digit from probabilities
## Requirements
- text
- tensorflow>=2.0.0
- numpy>=1.18.0
- matplotlib>=3.0.0

## Author
Collins Kimani - @collinskimani482-sudo
