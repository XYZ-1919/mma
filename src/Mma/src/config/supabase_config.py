import logging
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()



SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  

def get_db_connection(admin=False):  
    """Creates and returns a Supabase client. Handles potential errors."""
    try:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL or SUPABASE_KEY environment variables are not set.")

        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        return supabase
    except Exception as e:
        return None