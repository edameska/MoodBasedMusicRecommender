import cv2
import numpy as np
import tensorflow as tf
import time
from collections import Counter
import os

def detect_emotion(model_path='emotion_model.h5', duration=60, show_video=True):
    model = tf.keras.models.load_model(model_path)
    emotion_labels = ['Angry', 'Happy', 'Sad', 'Fear', 'Neutral']
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    detected_emotions = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=[0, -1])

            prediction = model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            detected_emotions.append(emotion)

            if show_video:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if show_video:
            cv2.imshow('Emotion Detection', frame)
            if (time.time() - start_time > duration) or (cv2.getWindowProperty('Emotion Detection', cv2.WND_PROP_VISIBLE) < 1):
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if time.time() - start_time > duration:
                break

    cap.release()
    cv2.destroyAllWindows()

    if detected_emotions:
        return Counter(detected_emotions).most_common(1)[0][0]
    else:
        return None
