from flask import Blueprint, render_template, request, redirect, url_for, session, flash
import bcrypt
from src.config.supabase_config import get_db_connection
from ..models.user import User


auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if user already exists
        existing_user = User.get_by_username(username)
        if existing_user:
            flash('Username already exists', 'error')
            return render_template('register.html', error='Username already taken')
        
        try:
            # Hash password
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            # Create user with hashed password
            user = User.create(username, hashed_password)
            
            if user:
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('auth.login'))
            else:
                flash('Registration failed', 'error')
                return render_template('register.html', error='Registration failed')
                
        except Exception as e:
           
            return render_template('register.html', error='Registration failed')
    
    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.get_by_username(username)
        
        if user and user.verify_password(password):
            session['username'] = username
            session['user_id'] = user.user_id
            return redirect(url_for('recommendation.upload_image'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))