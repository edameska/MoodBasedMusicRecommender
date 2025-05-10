import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load your dataset
data = pd.read_csv('fer2013.csv')
data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
data['image'] = data['pixels'].apply(lambda x: x.reshape((48,48)))

# Your emotion mapping
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
inv_emotion_map = {v:k for k,v in emotion_map.items()}

data['emotion_name'] = data['emotion'].map(emotion_map)

# List of allowed emotions
allowed_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to review and correct
for idx in range(len(data)):
    img = data.iloc[idx]['image']
    current_emotion = data.iloc[idx]['emotion_name']
    
    plt.imshow(img, cmap='gray')
    plt.title(f"Current: {current_emotion}")
    plt.axis('off')
    plt.show()
    
    print(f"Allowed emotions: {allowed_emotions}")
    new_label = input("Type new emotion (or ENTER to keep current): ").strip()
    
    if new_label:
        if new_label not in allowed_emotions:
            print("Invalid emotion. Skipping correction.")
        else:
            data.at[idx, 'emotion_name'] = new_label
            data.at[idx, 'emotion'] = inv_emotion_map[new_label]
    
    cont = input("Press ENTER to continue, or type 'q' to quit: ")
    if cont.lower() == 'q':
        break

data.drop(columns=['image', 'pixels']).to_csv('fer2013_corrected.csv', index=False)
