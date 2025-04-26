import matplotlib
matplotlib.use('TkAgg')  # interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns





data = pd.read_csv("fer2013.csv") 

# parse pixel values
data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))

# reshape to image size 
image_size = 48 
data['image'] = data['pixels'].apply(lambda x: x.reshape((image_size, image_size)))

# displays multiple images
#fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # 1 row, 5 columns

#Data normailization

emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

data['emotion'] = data['emotion'].map(emotion_map)

#remove rows where the emotion is 'Disgust' or 'Suprised' bcuz they are less represented
df_filtered = data[~data['emotion'].isin(['Disgust', 'Surprise'])]


# for i, emotion in enumerate(['Angry', 'Happy', 'Sad', 'Fear', 'Neutral']):
#     emotion_data = df_filtered[df_filtered['emotion'] == emotion]
#     first_image = emotion_data['image'].iloc[0]  #get the first image for each emotion
#     axes[i].imshow(first_image, cmap='gray')
#     axes[i].set_title(f"Emotion: {emotion}")
#     axes[i].axis('off')

# plt.tight_layout()
#plt.show()


# Count emotion occurrences
#sns.countplot(data=df_filtered, x='emotion')
#plt.title('Emotion Distribution')
#plt.show()


#dividing it to test and train data

train_data = df_filtered[df_filtered['Usage'] == 'Training']
test_data = df_filtered[df_filtered['Usage'].isin(['PublicTest', 'PrivateTest'])]

#train data distribution
#sns.countplot(data=train_data, x='emotion')
#plt.title('Emotion Distribution')
#plt.show()


#test data distribution
#sns.countplot(data=test_data, x='emotion')
#plt.title('Emotion Distribution')
#plt.show()

import tensorflow as tf
from tensorflow.keras import layers, models

# Prepare X and y
X_train = np.stack(train_data['image'].values) / 255.0  # normalize pixels
X_test = np.stack(test_data['image'].values) / 255.0

y_train = pd.get_dummies(train_data['emotion']).values
y_test = pd.get_dummies(test_data['emotion']).values

X_train = X_train.reshape(-1, 48, 48, 1)  # CNN needs channels
X_test = X_test.reshape(-1, 48, 48, 1)

# Build a simple CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Dropout layer to prevent overfitting
    layers.Dense(5, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 'adam' is the optimizer, which adjusts weights during training to make the model learn faster.
# 'categorical_crossentropy' is the loss function for multi-class classification, 
# measuring how far off the model's predictions are from the actual values.
# 'accuracy' is the metric used to track how many correct predictions the model makes.
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
model.save('emotion_model.h5')

#see which makes most mistakes
emotion_labels = ['Angry', 'Happy', 'Sad', 'Fear', 'Neutral']

from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()






