from flask import Flask, render_template
from .routes.auth_routes import auth_bp
from .routes.feedback_routes import feedback_bp
from .routes.recommendation_routes import recommendation_bp
from src.config.config import Config

import os

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.secret_key = Config.SECRET_KEY 

    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    

    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(feedback_bp, url_prefix='/feedback')
    app.register_blueprint(recommendation_bp, url_prefix='/recommendation')
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
   
    
    return app

