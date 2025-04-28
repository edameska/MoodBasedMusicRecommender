import cv2
import numpy as np
import tensorflow as tf
import time
from collections import Counter

# Load model
model = tf.keras.models.load_model('emotion_model.h5')
emotion_labels = ['Angry', 'Happy', 'Sad', 'Fear', 'Neutral']

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

# Track detected emotions
detected_emotions = []

# Timer
start_time = time.time()
duration = 60  # 1 minute

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w] #convert to grayscale and crop the face
        face = cv2.resize(face, (48, 48))# resize to 48x48
        face = face.astype('float32') / 255.0 # normalize pixel values
        face = np.expand_dims(face, axis=[0, -1])  # Shape (1, 48, 48, 1) because model expects 4D input

        prediction = model.predict(face, verbose=0) #verbose=0 to suppress output
        emotion = emotion_labels[np.argmax(prediction)]# get the index of the highest probability
        detected_emotions.append(emotion)# 

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # draw rectangle around face
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # put text above rectangle

    cv2.imshow('Emotion Detection', frame)

    # Stop after 1 minute or if window is closed
    if (time.time() - start_time > duration) or (cv2.getWindowProperty('Emotion Detection', cv2.WND_PROP_VISIBLE) < 1):
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Find and print the dominant emotion
if detected_emotions:
    dominant_emotion = Counter(detected_emotions).most_common(1)[0][0]
    print(f"Dominant emotion detected: {dominant_emotion}")
else:
    print("No emotions detected.")
