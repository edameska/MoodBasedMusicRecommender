import pandas as pd
import numpy as np
import sys
import os

# Import get_feature_ranges from your recommender module
from mood_util  import (
    get_feature_ranges,
    filter_songs_by_emotion,
    add_personal_match_scores,
    rank_recommendations,
    recommend_top_songs,
    add_mood_scores,
)


import pandas as pd
import numpy as np

def purity(recommendations_df, emotion):
    """
    Percentage of recommended songs within the emotion's audio feature ranges.
    """
    ranges = get_feature_ranges(emotion)
    count_within = 0
    total = len(recommendations_df)
    if total == 0:
        return 0.0
    
    for _, row in recommendations_df.iterrows():
        if all(ranges[feat][0] <= row[feat] <= ranges[feat][1] for feat in ranges):
            count_within += 1
    
    return 100.0 * count_within / total

def average_deviation(recommendations_df, emotion):
    """
    Average Euclidean distance of recommended songs' features from the centroid
    (midpoint of feature ranges) for the target emotion.
    """
    ranges = get_feature_ranges(emotion)
    if len(recommendations_df) == 0:
        return float('nan')
    
    # Calculate centroid (midpoint of each feature range)
    centroid = {feat: (low + high) / 2 for feat, (low, high) in ranges.items()}
    
    # For each song, calculate Euclidean distance from centroid using emotion features only
    distances = []
    for _, row in recommendations_df.iterrows():
        dist_sq = 0
        for feat in ranges:
            val = row[feat]
            dist_sq += (val - centroid[feat]) ** 2
        distances.append(np.sqrt(dist_sq))
    
    return float(np.mean(distances))

def evaluate_for_emotion(df, emotion, top_artists, top_tracks, top_n=10):
    # Filter songs by emotion
    filtered = filter_songs_by_emotion(df, emotion)
    
    # Add mood scores first (important!)
    scored = add_mood_scores(filtered, emotion)
    
    # Add personal match scores
    scored = add_personal_match_scores(scored, top_artists, top_tracks)
    
    # Rank recommendations by combined score
    ranked = rank_recommendations(scored)
    
    # Pick top N recommendations
    recommendations = recommend_top_songs(ranked, top_n)
    
    print(f"\nTop {top_n} recommendations for emotion '{emotion}':")
    print(recommendations[['Title', 'Artists', 'final_score']].to_string(index=False))
    
    # Compute metrics
    purity_score = purity(recommendations, emotion)
    avg_dev = average_deviation(recommendations, emotion)
    songs_returned = len(recommendations)
    
    return {
        "Emotion": emotion,
        "Songs Returned": songs_returned,
        "Purity (%)": round(purity_score, 2),
        "Avg Deviation": round(avg_dev, 4) if not np.isnan(avg_dev) else None
    }

    
    return {
        "Emotion": emotion,
        "Songs Returned": songs_returned,
        "Purity (%)": round(purity_score, 2),
        "Avg Deviation": round(avg_dev, 4) if not np.isnan(avg_dev) else None
    }

def evaluate_all_emotions(df, top_artists, top_tracks, emotions=None, top_n=10):
    if emotions is None:
        emotions = ["Angry", "Fear", "Happy", "Neutral", "Sad"]
    results = []
    for emotion in emotions:
        result = evaluate_for_emotion(df, emotion, top_artists, top_tracks, top_n)
        results.append(result)
    return pd.DataFrame(results)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth

    # Load environment variables for Spotify API
    load_dotenv()
    
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        redirect_uri="http://localhost:8888/callback",
        scope="user-top-read"
    ))
    
    # Load your big dataset
    df = pd.read_csv('Spotify_Dataset_V3.csv', delimiter=';')
    
    # Get user top artists and tracks
    top_tracks_data = sp.current_user_top_tracks(limit=20, time_range='medium_term')
    top_artists_data = sp.current_user_top_artists(limit=20, time_range='medium_term')
    top_tracks = [item['name'] for item in top_tracks_data['items']]
    top_artists = [item['name'] for item in top_artists_data['items']]
    
    # Evaluate across all emotions
    results_df = evaluate_all_emotions(df, top_artists, top_tracks, top_n=10)
    
    print("\nPersonalized Music Recommender System Evaluation Summary (Case Study)")
    print(results_df.to_string(index=False))
