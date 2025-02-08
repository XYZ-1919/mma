"""
Base calculator module defining the interface for similarity calculations.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict

class BaseSimilarityCalculator(ABC):
    """Abstract base class for similarity calculations."""
    
    @abstractmethod
    def calculate_similarity(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate similarity matrix from features.
        
        Args:
            features: Array of features to calculate similarities from
            
        Returns:
            Similarity matrix as numpy array
        """
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """List of feature names used by this calculator."""
        pass
    
    @abstractmethod
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess features before similarity calculation.
        
        Args:
            features: Raw feature array
            
        Returns:
            Preprocessed feature array
        """
        pass

class SimilarityException(Exception):
    """Base exception class for similarity calculation errors."""
    pass

class FeaturePreprocessingError(SimilarityException):
    """Raised when feature preprocessing fails."""
    pass

class SimilarityCalculationError(SimilarityException):
    """Raised when similarity calculation fails."""
    pass
