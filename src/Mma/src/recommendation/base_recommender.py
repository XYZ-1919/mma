from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseRecommender(ABC):
    def __init__(self):
        self.songs_df = None
        self.similarity_matrix = None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def get_recommendations(self, user_id, emotion, meta_features=None):
        pass

    def _normalize_features(self, features):
        return (features - features.min()) / (features.max() - features.min())

    def _get_mood_weights(self, emotion):
        mood_weights = {
            'happy': {'happy_score': 0.6, 'energetic_score': 0.2, 'neutral_score': 0.1},
            'sad': {'sad_score': 0.6, 'neutral_score': 0.2},
            'angry': {'angry_score': 0.6, 'energetic_score': 0.2},
            'neutral': {'neutral_score': 0.4, 'happy_score': 0.2, 'sad_score': 0.2},
            'energetic': {'energetic_score': 0.6, 'happy_score': 0.2}
        }
        return mood_weights.get(emotion, {'neutral_score': 0.5})