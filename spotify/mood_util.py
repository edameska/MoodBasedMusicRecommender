import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv

df = pd.read_csv('Spotify_Dataset_V3.csv', delimiter=';')

def get_feature_ranges(emotion):
    if emotion == "Angry":
        return {
            "Valence": (0.1, 0.4),    # Low valence (negative emotion) [1]
            "Energy": (0.7, 1.0),     # High energy (arousal) [1]
            "Loudness": (-8, 0),    # Loudness correlates with aggression [2]
            "Speechiness": (0.2, 0.95) # Angry songs often have rap/aggressive vocals [3]
        }
    elif emotion == "Fear":
        return {
            "Valence": (0.0, 0.3),    # Very low valence (unpleasant) [1]
            "Energy": (0.4, 0.8),     # Moderate-high energy (tension) [4]
            "Instrumentalness": (0.5, 1.0) # Fear often uses instrumental tracks (e.g., horror scores) [5]
        }
    elif emotion == "Happy":
        return {
            "Valence": (0.7, 1.0),   # High valence (positive) [1]
            "Energy": (0.6, 1.0),     # High energy [1]
            "Danceability": (0.6, 1.0) # Happy music is often danceable [6]
        }
    elif emotion == "Sad":
        return {
            "Valence": (0.0, 0.4),    # Low valence [1]
            "Energy": (0.1, 0.5),     # Low energy [1]
            "Acousticness": (0.5, 1.0) # Sad songs often use acoustic instruments [7]
        }
    else:  # Neutral
        return {
            "Valence": (0.4, 0.6),   # Mid-range valence
            "Energy": (0.3, 0.6)      # Moderate energy
        }
def filter_songs_by_emotion(df, emotion):
    """
    Filter songs in df to those within the feature ranges of the given emotion.
    """
    ranges = get_feature_ranges(emotion)
    filtered = df.drop_duplicates(subset=['Title', 'Artists'])
    for feature, (low, high) in ranges.items():
        filtered = filtered[(filtered[feature] >= low) & (filtered[feature] <= high)]
    return filtered

def add_personal_match_scores(df, top_artists, top_songs):
    """
    Adds binary columns for artist and song matches based on user's top artists/songs.
    Handles multiple artists by splitting on comma.
    """
    df = df.copy()
    top_artists_set = set(a.lower() for a in top_artists)
    top_songs_set = set(s.lower() for s in top_songs)

    def artist_in_top(artist_str):
        artists = [a.strip().lower() for a in artist_str.split(',')]
        return int(any(a in top_artists_set for a in artists))

    df['artist_match'] = df['Artists'].apply(artist_in_top)
    df['song_match'] = df['Title'].str.lower().apply(lambda t: 1 if t in top_songs_set else 0)
    return df

def score_feature(value, low, high):
    if value < low or value > high:
        return 0
    mid = (low + high) / 2
    max_dist = (high - low) / 2
    dist = abs(value - mid)
    return 1 - (dist / max_dist)

def add_mood_scores(df, emotion):
    ranges = get_feature_ranges(emotion)
    df = df.copy()
    mood_scores = []
    for idx, row in df.iterrows():
        scores = []
        for feature, (low, high) in ranges.items():
            scores.append(score_feature(row[feature], low, high))
        mood_scores.append(sum(scores) / len(scores))
    df['mood_score'] = mood_scores
    return df

def rank_recommendations(df, mood_weight=0.5, artist_weight=0.25, song_weight=0.25):
    """
    Ranks songs by combining mood closeness, artist, and song match scores.
    Mood closeness currently constant; consider implementing a real score.
    """
    df = df.copy()
    df['final_score'] = (mood_weight * df['mood_score']) + (artist_weight * df['artist_match']) + (song_weight * df['song_match'])
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

    emotion = "Happy"  # Simulated detected emotion
    
    import sys
    import os
    # Add the parent directory and then mood_recognition to the path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mood_recognition')))

    from mood_detect import detect_emotion
    #emotion = detect_emotion(model_path="/home/edameska/Desktop/New Folder/graduation /mood_recognition/emotion_model.h5")
    print("Detected Emotion:", emotion)
   
    filtered_df = df.drop_duplicates(subset=['Title', 'Artists'])
    ranges = get_feature_ranges(emotion)

    # Debugging: print how many songs remain after each feature filter
    for feature, (low, high) in ranges.items():
        before = len(filtered_df)
        filtered_df = filtered_df[(filtered_df[feature] >= low) & (filtered_df[feature] <= high)]
        print(f"After filtering {feature} ({low}-{high}): {before} -> {len(filtered_df)} songs left")

    # Add mood scores
    scored_df = add_mood_scores(filtered_df, emotion)

    # Add personal match scores (artist and song)
    scored_df = add_personal_match_scores(scored_df, top_artists, top_tracks)

    # Rank by combined score
    ranked_df = rank_recommendations(scored_df)

    # Recommend top 10 songs
    recommendations = recommend_top_songs(ranked_df, n=10)

    print(f"\nTop 10 personalized songs for emotion: {emotion}")

    print(recommendations[['Title', 'Artists', 'final_score']])
    recommendations[['Title', 'Artists', 'final_score']].to_csv('full_recommendations.csv', index=False)
