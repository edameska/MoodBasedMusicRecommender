import pandas as pd


df = pd.read_csv('Spotify_Dataset_V3.csv', delimiter=';')

#group by Danceability

def classify_danceability(value):
    if value <= 0.5:
        return 'low'
    else:
        return 'high'


df['danceability_group'] = df['Danceability'].apply(classify_danceability)

#2 dataframes
low_danceability = df[df['danceability_group'] == 'low']
high_danceability = df[df['danceability_group'] == 'high']

#csv files
low_danceability.to_csv('low_danceability.csv', index=False)
high_danceability.to_csv('high_danceability.csv', index=False)
