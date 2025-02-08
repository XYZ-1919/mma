import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

class Config:
    # Supabase
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
    
    # API Keys
    WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY')
    
    # Application
    UPLOAD_FOLDER = 'src/static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MODEL_PATH = os.getenv('MODEL_PATH')

    @staticmethod
    def get_db_connection(admin=False):
        try:
            if not all([Config.SUPABASE_URL, Config.SUPABASE_KEY]):
                raise ValueError("Missing Supabase credentials")
            return create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        except Exception:
            return None
