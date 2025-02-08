from flask import Blueprint, request, session, jsonify
from ..models.feedback import Feedback
from datetime import datetime
from ..utils.decorators import require_auth, handle_errors

feedback_bp = Blueprint('feedback', __name__)

@feedback_bp.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        user_id = session['user_id']
        track_id = request.json.get('track_id')
        rating = request.json.get('rating')
        mood = session.get('emotion') or request.json.get('mood')
        if not all([track_id, rating, mood]):
            return jsonify({"error": "Missing required parameters"}), 400
        if Feedback.create(user_id, track_id, rating, mood):
            return jsonify({
                "success": True,
                "data": {
                    "user_id": user_id,
                    "track_id": track_id,
                    "rating": rating,
                    "mood": mood
                }
            }), 200
        return jsonify({"error": "Failed to submit feedback"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@feedback_bp.route('/get_feedback', methods=['GET'])
def get_feedback():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        user_id = session['user_id']
        mood = request.args.get('mood')
        feedback = Feedback.get_user_feedback(user_id, mood)
        return jsonify(feedback.to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500