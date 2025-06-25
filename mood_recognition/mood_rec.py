import matplotlib
matplotlib.use('TkAgg')  # interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Load and preprocess data
data = pd.read_csv("final_merged_file.csv") 

# Parse pixel values
data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))

# Reshape to image size 
image_size = 48 
data['image'] = data['pixels'].apply(lambda x: x.reshape((image_size, image_size)))

# Data normalization and emotion mapping
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
data['emotion'] = data['emotion'].map(emotion_map)

# Remove under-represented classes
df_filtered = data[~data['emotion'].isin(['Disgust', 'Surprise'])]

# Split data
train_data = df_filtered[df_filtered['Usage'] == 'Training']
test_data = df_filtered[df_filtered['Usage'].isin(['PublicTest', 'PrivateTest'])]


# Prepare X and y
X_train = np.stack(train_data['image'].values) / 255.0
X_test = np.stack(test_data['image'].values) / 255.0
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)
y_train = pd.get_dummies(train_data['emotion']).values
y_test = pd.get_dummies(test_data['emotion']).values


X_val, X_final_test, y_val, y_final_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)




# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced',
                                                classes=np.unique(np.argmax(y_train, axis=1)),
                                                y=np.argmax(y_train, axis=1))
class_weights = dict(enumerate(class_weights))

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build model
model = models.Sequential()

# First Conv block
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
# 32 filters (feature detectors) of size 3x3, ReLU activation for non-linearity
# 'same' padding preserves spatial dimensions, input_shape matches our 48x48 grayscale images
model.add(layers.BatchNormalization())# Normalizes activations to stabilize and accelerate training
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# MaxPooling reduces spatial dimensions, retaining important features
# 2x2 pooling size reduces each feature map by half
model.add(layers.Dropout(0.25))
# Dropout regularization to prevent overfitting, randomly dropping 25% of neurons

# Second Conv block
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
# 64 filters for more complex feature extraction
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

# Third Conv block
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
# 128 filters for deeper feature extraction
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

# Flatten and dense layers
# Flattening the 3D output to 1D for dense layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))  # Reduced from 1024
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
# Higher dropout (50%) as dense layers are more prone to overfitting
model.add(layers.Dense(256, activation='relu'))  # Additional layer for better learning
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(5, activation='softmax'))  # Explicit 5 classes
# Softmax converts outputs to probability distribution across classes


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), 
    loss='categorical_crossentropy',
    metrics=['accuracy'] 
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),# Early stopping to prevent overfitting
    # Stops training if validation loss doesn't improve for 10 epochs
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001),# Reduces learning rate if validation loss plateaus
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Train with augmented data
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=64),
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weights
)

# Save model
model.save('emotion_model.h5')

# Evaluation
emotion_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']  # Remaining classes after filtering
 # Evaluate on final test set (not validation set)
test_loss, test_accuracy = model.evaluate(X_final_test, y_final_test, verbose=0)
print(f"Final Test Accuracy: {test_accuracy:.4f}")

# Confusion matrix on final test set
y_pred = np.argmax(model.predict(X_final_test), axis=1)
y_true = np.argmax(y_final_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', 
            xticklabels=emotion_labels, 
            yticklabels=emotion_labels)
plt.title('Confusion Matrix - Final Test Set')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=emotion_labels))

# Plot training history
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

# Optional: Plot learning rate schedule
if 'lr' in history.history:
    plt.figure(figsize=(8,4))
    plt.plot(history.history['lr'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.show()