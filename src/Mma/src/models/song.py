from src.config.supabase_config import get_db_connection
import pandas as pd
from sqlalchemy import text

class Song:
    def __init__(self, song_data):
        self.data = song_data

    @classmethod
    def get_songs_by_emotion(cls, emotion):
        """Retrieve songs for a specific emotion"""
        try:
            supabase = get_db_connection()
            response = supabase.table('songs').select('*').eq('predicted_mood', emotion).limit(20).execute()
            
            if response.data:
                songs_df = pd.DataFrame(response.data)
                return songs_df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error retrieving songs: {e}")
            return pd.DataFrame()

    @classmethod
    def get_song_by_track_id(cls, track_id):
        """Retrieve a specific song by track ID"""
        engine = get_db_connection()
        try:
            query = text("SELECT * FROM songs WHERE track_id = :track_id")
            df = pd.read_sql(query, engine, params={'track_id': track_id})
            return df.iloc[0].to_dict() if not df.empty else None
        except Exception as e:
            print(f"Error retrieving song: {e}")
            return None