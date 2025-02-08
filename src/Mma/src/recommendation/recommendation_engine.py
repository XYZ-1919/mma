import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from ..config.config import Config
from .meta_features import MetaFeatureExtractor
from .base_recommender import BaseRecommender
from scipy.sparse.linalg import svds
from scipy.stats import beta
import requests
import os
class RecommendationEngine(BaseRecommender):
    def __init__(self, weather_api_key):
        super().__init__()
        self.meta_feature_extractor = MetaFeatureExtractor(weather_api_key)
        self.load_data()
        self.preprocess_data()
        self.user_factors = None
        self.item_factors = None

    def load_data(self):
        supabase = Config.get_db_connection()
        
        # Fetch data from Supabase
        songs_response = supabase.table('songs').select("*").execute()
        feedback_response = supabase.table('feedback').select("*").execute()
        history_response = supabase.table('recommendation_history').select("*").execute()
        
        # Convert to pandas DataFrames
        self.songs_df = pd.DataFrame(songs_response.data)
        self.feedback_df = pd.DataFrame(feedback_response.data)
        self.recommendation_history = pd.DataFrame(history_response.data)

    def preprocess_data(self):
        # Normalize numerical features
        numerical_features = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'tempo', 'emotional_intensity', 'word_complexity'
        ]
        self.songs_df[numerical_features] = self.songs_df[numerical_features].apply(
            self._normalize_features
        )

    def _apply_meta_feature_weights(self, scores, meta_features):
        if not meta_features:
            return scores
            
        try:
            # Weather-based adjustments
            weather = meta_features.get('weather_condition', 'unknown')
            weather_weights = {
                'Clear': {'happy_score': 0.1, 'energetic_score': 0.1},
                'Rain': {'sad_score': 0.1, 'acousticness': 0.1},
                'Snow': {'acousticness': 0.15, 'instrumentalness': 0.1},
                'Thunderstorm': {'energy': 0.15, 'loudness': 0.1}
            }

            # Time-based adjustments
            time_of_day = meta_features.get('time_of_day', 'unknown')
            time_weights = {
                'morning': {'energy': 0.15, 'valence': 0.1},
                'afternoon': {'danceability': 0.15, 'energy': 0.1},
                'evening': {'instrumentalness': 0.1, 'acousticness': 0.1},
                'night': {'acousticness': 0.15, 'instrumentalness': 0.15}
            }

            # Apply weather weights
            if weather in weather_weights:
                for feature, weight in weather_weights[weather].items():
                    if feature in self.songs_df.columns:
                        scores += self.songs_df[feature] * weight

            # Apply time weights
            if time_of_day in time_weights:
                for feature, weight in time_weights[time_of_day].items():
                    scores += self.songs_df[feature] * weight

            return scores

        except Exception as e:
            return scores

    def _get_user_preferences(self, user_id, emotion):
        """Enhanced user preferences with temporal dynamics"""
        if self.feedback_df.empty:
            return None
            
        user_feedback = self.feedback_df[self.feedback_df['user_id'] == user_id]
        if user_feedback.empty:
            return None
            
        # Add temporal weighting
        current_time = pd.Timestamp.now()
        user_feedback['time_weight'] = user_feedback['created_at'].apply(
            lambda x: np.exp(-0.1 * (current_time - pd.Timestamp(x)).days)
        )
        
        # Weight feedback by recency
        weighted_feedback = user_feedback.copy()
        weighted_feedback['rating'] *= weighted_feedback['time_weight']
        
        # Get positively rated songs with their features
        positive_songs = self.songs_df[
            self.songs_df['track_id'].isin(
                weighted_feedback[weighted_feedback['rating'] > 0]['track_id']
            )
        ]
        
        if not positive_songs.empty:
            # Calculate weighted feature preferences
            weighted_prefs = positive_songs.multiply(
                weighted_feedback[weighted_feedback['rating'] > 0]['rating'].values,
                axis=0
            ).mean()
            return weighted_prefs
        return None

    def _matrix_factorization(self):
        try:
            # Create user-item matrix
            user_item_matrix = self.feedback_df.pivot(
                index='user_id', 
                columns='track_id', 
                values='rating'
            ).fillna(0)
            
            # Ensure minimum dimensions and handle empty matrix
            if user_item_matrix.empty:
                self.user_factors = None
                self.item_factors = None
                return
                
            # Determine appropriate rank for SVD
            min_dim = min(user_item_matrix.shape[0], user_item_matrix.shape[1])
            k = min(min_dim - 1, 20) if min_dim > 1 else 1
            
            # Perform SVD
            U, sigma, Vt = svds(user_item_matrix.values, k=k)
            self.user_factors = U @ np.diag(sigma)
            self.item_factors = Vt.T
        except Exception as e:
            self.user_factors = None
            self.item_factors = None

    def _calculate_bayesian_rating(self, positive_ratings, total_ratings):
        """
        Calculate Bayesian average rating to handle uncertainty in small sample sizes.
        
        Args:
            positive_ratings (int): Number of positive ratings
            total_ratings (int): Total number of ratings
            
        Returns:
            float: Bayesian average rating between 0 and 1
        """
        # Using Beta distribution for rating uncertainty
        a = positive_ratings + 1  # Add 1 for Laplace smoothing
        b = total_ratings - positive_ratings + 1
        return beta.mean(a, b)
    
    def get_recommendations(self, user_id, emotion, n_recommendations=10):
        try:
            meta_features = self.meta_feature_extractor.get_all_meta_features()
            user_prefs = self._get_user_preferences(user_id, emotion)
            
            # Get base emotion-matched songs
            emotion_songs = self.songs_df[
                self.songs_df['predicted_mood'] == emotion
            ]
            
            if user_prefs is not None:
                # Calculate similarity to user preferences
                pref_scores = emotion_songs[self.feature_names].apply(
                    lambda x: cosine_similarity(
                        [x],
                        [user_prefs[self.feature_names]]
                    )[0][0],
                    axis=1
                )
                
                # Combine with collaborative filtering scores
                if self.user_factors is not None:
                    cf_scores = self._get_collaborative_scores(user_id, emotion_songs)
                    final_scores = 0.6 * pref_scores + 0.4 * cf_scores
                else:
                    final_scores = pref_scores
                
                # Apply meta-feature weights
                final_scores = self._apply_meta_feature_weights(final_scores, meta_features)
                
                # Get top recommendations
                top_indices = final_scores.nlargest(n_recommendations).index
                recommendations = emotion_songs.loc[top_indices].to_dict('records')
            else:
                recommendations = self._get_emotion_based_recommendations(
                    emotion, n_recommendations
                )
            
            return recommendations, meta_features
            
        except Exception as e:
            return self._get_emotion_based_recommendations(
                emotion, n_recommendations
            ), None
            
    def _get_collaborative_scores(self, user_id, candidate_songs):
        """Calculate collaborative filtering scores for candidate songs"""
        if self.user_factors is None or self.item_factors is None:
            return pd.Series(0, index=candidate_songs.index)
            
        user_vector = self.user_factors[user_id]
        song_indices = [
            self.songs_df[self.songs_df['track_id'] == tid].index[0]
            for tid in candidate_songs['track_id']
        ]
        return pd.Series(
            user_vector @ self.item_factors[song_indices].T,
            index=candidate_songs.index
        )

    def _get_emotion_based_recommendations(self, emotion, n_recommendations):
        """Fallback method with better error handling"""
        try:
            if self.songs_df.empty:
                return []
                
            emotion_songs = self.songs_df[
                self.songs_df['predicted_mood'] == emotion
            ]
            
            if emotion_songs.empty:
                return self.songs_df.sample(
                    n=min(n_recommendations, len(self.songs_df))
                ).to_dict('records')
                
            return emotion_songs.sample(
                n=min(n_recommendations, len(emotion_songs))
            ).to_dict('records')
            
        except Exception as e:
            return []

    def get_average_similarity_score(self, recommendations):
        """Calculate the average similarity score for the recommended songs"""
        try:
            if not recommendations:
                return 0.0
                
            # Extract feature columns used for similarity calculation
            feature_cols = ['danceability', 'energy', 'valence', 'tempo', 
                          'acousticness', 'instrumentalness']
            
            # Get features for recommended songs
            recommended_features = self.songs_df[
                self.songs_df['track_id'].isin([song['track_id'] for song in recommendations])
            ][feature_cols].values
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(recommended_features)
            
            # Calculate average similarity (excluding self-similarity)
            n = similarities.shape[0]
            if n <= 1:
                return 0.0
                
            # Mask the diagonal (self-similarities) and calculate mean
            np.fill_diagonal(similarities, 0)
            avg_similarity = similarities.sum() / (n * (n - 1))
            
            return float(avg_similarity * 100)  # Convert to percentage
            
        except Exception as e:
            return 0.0
