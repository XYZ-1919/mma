"""
Main similarity calculator combining audio and sentiment similarities.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from .matern_calculator import AudioFeatureCalculator, SentimentFeatureCalculator
from .base_calculator import SimilarityException, BaseSimilarityCalculator


class SongSimilarityCalculator:
    """Main class for calculating song similarities."""
    
    def __init__(self, 
                 cache_dir: str = 'cache/similarity',
                 audio_weight: float = 0.6,
                 sentiment_weight: float = 0.4):
        """
        Initialize the similarity calculator.
        
        Args:
            cache_dir: Directory for caching similarity matrices
            audio_weight: Weight for audio features
            sentiment_weight: Weight for sentiment features
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.audio_weight = audio_weight
        self.sentiment_weight = sentiment_weight
        
        self.audio_calculator = AudioFeatureCalculator()
        self.sentiment_calculator = SentimentFeatureCalculator()
    
    def _get_cache_path(self, mood_label: str) -> Path:
        """Get path for cached similarity matrix."""
        return self.cache_dir / f"similarity_matrix_{mood_label}.joblib"
    
    def _extract_features(self, 
                         data: pd.DataFrame, 
                         calculator: BaseSimilarityCalculator) -> np.ndarray:
        """Extract features for a specific calculator."""
        return data[calculator.feature_names].values
    
    def calculate_mood_similarity(self, 
                                mood_songs: pd.DataFrame,
                                force_recalculate: bool = False) -> np.ndarray:
        """
        Calculate similarity matrix for songs within a mood cluster.
        
        Args:
            mood_songs: DataFrame containing songs from a mood cluster
            force_recalculate: Force recalculation even if cached
            
        Returns:
            Similarity matrix as numpy array
        """
        mood_label = mood_songs['predicted_mood'].iloc[0]
        cache_path = self._get_cache_path(mood_label)
        
        # Try to load from cache unless force_recalculate is True
        if not force_recalculate and cache_path.exists():
            try:
                return joblib.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cached similarity matrix: {e}")
        
        try:
            # Calculate audio similarity
            audio_features = self._extract_features(mood_songs, self.audio_calculator)
            audio_similarity = self.audio_calculator.calculate_similarity(audio_features)
            
            # Calculate sentiment similarity
            sentiment_features = self._extract_features(mood_songs, self.sentiment_calculator)
            sentiment_similarity = self.sentiment_calculator.calculate_similarity(sentiment_features)
            
            # Combine similarities with weights
            combined_similarity = (
                self.audio_weight * audio_similarity +
                self.sentiment_weight * sentiment_similarity
            )
            
            # Cache the result
            joblib.dump(combined_similarity, cache_path)
            
            return combined_similarity
            
        except Exception as e:
           
            raise SimilarityException(f"Failed to calculate similarity matrix: {e}")
    
    def get_similar_songs(self,
                         mood_songs: pd.DataFrame,
                         track_id: Optional[str] = None,
                         n_recommendations: int = 15) -> Tuple[List[Dict], np.ndarray]:
        """
        Get similar songs based on similarity matrix.
        
        Args:
            mood_songs: DataFrame containing songs from a mood cluster
            track_id: Reference track ID (optional)
            n_recommendations: Number of recommendations to return
            
        Returns:
            Tuple containing:
            - List of recommended songs as dictionaries
            - Similarity matrix
        """
        try:
            similarity_matrix = self.calculate_mood_similarity(mood_songs)
            
            if track_id:
                song_idx = mood_songs.index[mood_songs['track_id'] == track_id].tolist()[0]
                similarities = similarity_matrix[song_idx]
            else:
                # If no reference track, use aggregate similarity
                similarities = similarity_matrix.mean(axis=0)
            
            # Get top recommendations
            top_indices = similarities.argsort()[::-1][:n_recommendations]
            recommendations = mood_songs.iloc[top_indices]
            
            return recommendations.to_dict('records'), similarity_matrix
            
        except Exception as e:
            raise SimilarityException(f"Failed to get similar songs: {e}")
