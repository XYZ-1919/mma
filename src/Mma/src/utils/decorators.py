from functools import wraps
from flask import session, redirect, url_for, render_template

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated

def handle_errors(template):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                return render_template(template, error_message=str(e))
        return decorated
    return decorator
