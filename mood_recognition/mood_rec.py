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




# Calculate class weights (keep this part the same)
class_weights = class_weight.compute_class_weight('balanced',
                                                classes=np.unique(np.argmax(y_train, axis=1)),
                                                y=np.argmax(y_train, axis=1))
class_weights = dict(enumerate(class_weights))

# Enhanced data augmentation (keep this the same)
train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build model according to data dimensions (48x48x1)
model = models.Sequential()

# Input layer - matches data (48x48x1)
model.add(layers.Input(shape=(48, 48, 1)))

# First Conv block (48x48x32)
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # Reduces to 24x24

# Second Conv block (24x24x64)
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # Reduces to 12x12

# Third Conv block (12x12x128)
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # Reduces to 6x6

# Fourth Conv block (6x6x256)
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # Reduces to 3x3

# Fifth Conv block (3x3x64)
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())

# Flatten and dense layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))  # 5 classes as per your filtered data

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), 
    loss='categorical_crossentropy',
    metrics=['accuracy'] 
)

model.summary()

# Callbacks (keep these the same)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001),
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

# Plot learning rate schedule
if 'lr' in history.history:
    plt.figure(figsize=(8,4))
    plt.plot(history.history['lr'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.show()