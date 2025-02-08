from flask import Blueprint, render_template, request, session, redirect, url_for
from PIL import Image
from io import BytesIO
from ..recommendation.recommendation_engine import RecommendationEngine
from ..emotion_detection.model import EmotionDetector
from ..utils.decorators import require_auth, handle_errors
from src.config.config import Config

recommendation_bp = Blueprint('recommendation', __name__)
recommender = RecommendationEngine(weather_api_key=Config.WEATHER_API_KEY)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@recommendation_bp.route('/upload_image', methods=['GET', 'POST'])
@require_auth
@handle_errors('upload_image.html')
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if not file or not allowed_file(file.filename):
            return render_template('upload_image.html', error_message='Invalid file')
        img = Image.open(BytesIO(file.read()))
        emotion, confidence = EmotionDetector().detect_emotion(img)
        recommendations, meta_features = recommender.get_recommendations(
            user_id=session['user_id'],
            emotion=emotion
        )
        session['emotion'] = emotion
        session['recommendation_ids'] = [song['track_id'] for song in recommendations]
        metrics = {
            'emotion_confidence': confidence,
            'similarity_score': recommender.get_average_similarity_score(recommendations),
            'total_features': len(recommender.songs_df.columns) * len(recommendations)
        }
        return render_template('dashboard.html',
                           emotion=emotion,
                           songs=recommendations,
                           meta_features=meta_features,
                           metrics=metrics)
    return render_template('upload_image.html')

@recommendation_bp.route('/refresh_recommendations', methods=['GET', 'POST'])
def refresh_recommendations():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    user_id = session['user_id']
    emotion = session.get('emotion') or request.args.get('emotion')
    if not emotion:
        return redirect(url_for('recommendation.upload_image'))
    session['emotion'] = emotion
    default_meta_features = {
        'weather_condition': 'Unknown',
        'temperature': 'N/A',
        'time_of_day': 'N/A',
        'city': 'Unknown',
        'country': 'Unknown'
    }
    try:
        recommendations, meta_features = recommender.get_recommendations(
            user_id=user_id,
            emotion=emotion
        )
        meta_features = meta_features or default_meta_features
        if not recommendations:
            return render_template('dashboard.html',
                                emotion=emotion,
                                error_message="Unable to get new recommendations",
                                meta_features=default_meta_features,
                                songs=[],
                                metrics={'emotion_confidence': 0, 'similarity_score': 0, 'total_features_count': 0})
        for song in recommendations:
            song['spotify_link'] = f"https://open.spotify.com/track/{song['track_id']}"
        session['recommendation_ids'] = [song['track_id'] for song in recommendations]
        metrics = {
            'emotion_confidence': session.get('emotion_confidence', 0.85),
            'similarity_score': recommender.get_average_similarity_score(recommendations),
            'total_features_count': len(recommender.songs_df.columns) * len(recommendations)
        }
        return render_template('dashboard.html',
                            emotion=emotion,
                            songs=recommendations,
                            meta_features=meta_features,
                            metrics=metrics)
    except Exception:
        return render_template('dashboard.html',
                            emotion=session.get('emotion', ''),
                            error_message="Error getting recommendations",
                            meta_features=default_meta_features,
                            songs=[],
                            metrics={'emotion_confidence': 0, 'similarity_score': 0, 'total_features_count': 0})
