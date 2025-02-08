from ..config.config import Config
import pandas as pd

class Feedback:
    def __init__(self, user_id, track_id, rating, mood=None):
        self.user_id = user_id
        self.track_id = track_id
        self.rating = rating
        self.mood = mood

    @classmethod
    def create(cls, user_id, track_id, rating, mood=None):
        try:
            supabase = Config.get_db_connection()
            response = supabase.table('feedback').insert({
                'user_id': user_id,
                'track_id': track_id,
                'rating': rating,
                'mood': mood
            }).execute()
            return True
        except Exception:
            return False

    @classmethod
    def get_user_feedback(cls, user_id, mood=None):
        try:
            supabase = Config.get_db_connection()
            query = supabase.table('feedback').select('*').eq('user_id', user_id)
            if mood:
                query = query.eq('mood', mood)
            response = query.execute()
            return pd.DataFrame(response.data)
        except Exception:
            return pd.DataFrame()