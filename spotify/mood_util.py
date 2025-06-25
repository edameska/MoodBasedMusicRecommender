import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv

df = pd.read_csv('Spotify_Dataset_V3.csv', delimiter=';')


def get_feature_ranges(emotion):
    if emotion == "Happy":
        return {"Valence": (0.7, 1.0), "Energy": (0.7, 1.0), "Danceability": (0.6, 1.0), "Acousticness": (0.0, 0.4), "Instrumentalness": (0.0, 0.2)}
    elif emotion == "Sad":
        return {"Valence": (0.0, 0.4), "Energy": (0.0, 0.5), "Danceability": (0.0, 0.5), "Acousticness": (0.6, 1.0), "Instrumentalness": (0.2, 0.8)}
    elif emotion == "Angry":
        return {"Valence": (0.0, 0.4), "Energy": (0.7, 1.0), "Danceability": (0.4, 0.7), "Acousticness": (0.0, 0.3), "Instrumentalness": (0.0, 0.3)}
    elif emotion == "Fear":
        return {"Valence": (0.0, 0.3), "Energy": (0.4, 0.6), "Danceability": (0.0, 0.5), "Acousticness": (0.5, 1.0), "Instrumentalness": (0.3, 1.0)}
    else:
        return {"Valence": (0.4, 0.6), "Energy": (0.4, 0.6), "Danceability": (0.4, 0.6), "Acousticness": (0.3, 0.6), "Instrumentalness": (0.0, 0.4)} #maybe change this later

# Filter songs based on emotion feature ranges
def filter_songs_by_emotion(df, emotion):
    ranges = get_feature_ranges(emotion)
    filtered = df.copy()
    filtered = filtered.drop_duplicates(subset=['Title', 'Artists'])
    for feature, (low, high) in ranges.items():
        filtered = filtered[(filtered[feature] >= low) & (filtered[feature] <= high)]
    return filtered

def add_personal_match_scores(df, top_artists, top_songs):
    df = df.copy()
    # Boost if artist in user top artists
    df['artist_match'] = df['Artists'].apply(lambda a: 1 if a in top_artists else 0)
    # Boost if song in user top songs
    df['song_match'] = df['Title'].apply(lambda t: 1 if t in top_songs else 0)
    return df
def rank_recommendations(df, mood_weight=0.5, artist_weight=0.2, song_weight=0.3):
    df = df.copy()
    # Combine weights for final ranking score
    df['final_score'] = (mood_weight * 1) + (artist_weight * df['artist_match']) + (song_weight * df['song_match'])
    return df.sort_values(by='final_score', ascending=False)

# Recommend top N songs 
def recommend_top_songs(df, n=10):
    return df.head(n)

if __name__ == "__main__":

    load_dotenv()

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        redirect_uri="http://localhost:8888/callback",
        scope="user-top-read"
    ))
    
    # Get user top artists and tracks
    top_tracks_data = sp.current_user_top_tracks(limit=20, time_range='medium_term')
    top_artists_data = sp.current_user_top_artists(limit=20, time_range='medium_term')
    
    top_tracks = [item['name'] for item in top_tracks_data['items']]
    top_artists = [item['name'] for item in top_artists_data['items']]

    #emotion = "Angry"  # Simulated detected emotion
    
    import sys
    import os
    # Add the parent directory and then mood_recognition to the path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mood_recognition')))

    from mood_detect import detect_emotion
    emotion = detect_emotion(model_path="/home/edameska/Desktop/New Folder/graduation /mood_recognition/emotion_model.h5")
    print("Detected Emotion:", emotion)
   
    filtered_df = filter_songs_by_emotion(df, emotion)
    # Add personal match scores based on user data
    scored_df = add_personal_match_scores(filtered_df, top_artists, top_tracks)
    # Rank by combined score
    ranked_df = rank_recommendations(scored_df)
    # Pick top N
    recommendations = recommend_top_songs(ranked_df, n=10)

    print(f"\nTop 10 personalized songs for emotion: {emotion}")
    print(recommendations[['Title', 'Artists', 'final_score']])
