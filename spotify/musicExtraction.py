from dotenv import load_dotenv
import os 
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

print(client_id,client_secret)

#authorisation

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri="http://localhost/",
                                               scope=["user-top-read", "user-library-read"]))


results = sp.current_user_top_tracks(limit=20, time_range='short_term')  


print("Your top 20 tracks:")
for idx, item in enumerate(results['items']):
    print(f"{idx+1}. {item['name']} by {item['artists'][0]['name']}")
    track_metadata = sp.track(item['id'])  
    print(track_metadata)

with open("top_tracks.txt", "w") as file:
    file.write("Your top 20 tracks:\n")
    for idx, item in enumerate(results['items']):
        file.write(f"{idx+1}. {item['name']} by {item['artists'][0]['name']}\n")
        track_metadata = sp.track(item['id'])
        file.write(f"  Track metadata: {track_metadata}\n")