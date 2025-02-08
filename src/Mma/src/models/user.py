from ..config.config import Config
import bcrypt

class User:
    def __init__(self, username=None, password=None, user_id=None):
        self.username = username
        self.password = password
        self.user_id = user_id

    @classmethod
    def get_by_username(cls, username):
        try:
            supabase = Config.get_db_connection(admin=True)
            response = supabase.table('users').select("*").eq('username', username).execute()
            if response.data and len(response.data) > 0:
                user_data = response.data[0]
                return cls(
                    username=user_data.get('username'),
                    password=user_data.get('password'),
                    user_id=user_data.get('user_id')
                )
            return None
        except Exception:
            return None

    @classmethod
    def create(cls, username, password):
        try:
            supabase = Config.get_db_connection(admin=True)
            existing = supabase.table('users').select("*").eq('username', username).execute()
            if existing.data:
                return None
            if isinstance(password, bytes):
                password = password.decode('utf-8')
            response = supabase.table('users').insert({
                'username': username,
                'password': password
            }).execute()
            return cls.get_by_username(username)
        except Exception:
            return None

    def verify_password(self, provided_password):
        try:
            if not self.password:
                return False
            if isinstance(provided_password, str):
                provided_password = provided_password.encode('utf-8')
            stored_hash = self.password
            if isinstance(stored_hash, str):
                stored_hash = stored_hash.encode('utf-8')
            return bcrypt.checkpw(provided_password, stored_hash)
        except Exception:
            return False