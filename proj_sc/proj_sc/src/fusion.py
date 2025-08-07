import numpy as np
from typing import List, Optional

class FeatureFusion:
    def __init__(self, normalize: bool = True, norm_type: str = 'l2', weights: Optional[List[float]] = None):
        """
        Initialize the feature fusion module.
        :param normalize: Whether to normalize the fused feature vector.
        :param norm_type: Type of normalization ('l2' or 'minmax').
        :param weights: Optional list of weights for each feature vector.
        """
        self.normalize = normalize
        self.norm_type = norm_type
        self.weights = weights

    def fuse(self, features: List[np.ndarray]) -> np.ndarray:
        """
        Concatenate and optionally normalize feature vectors.
        :param features: List of feature vectors (NumPy arrays) to fuse.
        :return: Fused feature vector (NumPy array).
        """
        fused = self._concatenate(features)
        if self.normalize:
            fused = self._normalize(fused)
        return fused

    def _concatenate(self, features: List[np.ndarray]) -> np.ndarray:
        """
        Concatenate feature vectors, applying weights if provided.
        :param features: List of feature vectors.
        :return: Concatenated feature vector.
        """
        if self.weights:
            if len(self.weights) != len(features):
                raise ValueError("Length of weights must match number of feature vectors.")
            weighted = [f * w for f, w in zip(features, self.weights)]
            return np.concatenate(weighted)
        return np.concatenate(features)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize the feature vector.
        :param vector: Feature vector.
        :return: Normalized feature vector.
        """
        if self.norm_type == 'l2':
            norm = np.linalg.norm(vector)
            return vector / norm if norm > 0 else vector
        elif self.norm_type == 'minmax':
            min_v, max_v = np.min(vector), np.max(vector)
            return (vector - min_v) / (max_v - min_v) if max_v > min_v else vector
        else:
            raise ValueError(f"Unsupported normalization type: {self.norm_type}")

# Example usage:
# import numpy as np
# from src.fusion import FeatureFusion
#
# # Suppose you have three feature vectors from POPMV, DWT, and color stats
# popmv_feat = np.random.rand(16)
# dwt_feat = np.random.rand(7)
# color_feat = np.random.rand(9)
#
# fusion = FeatureFusion(normalize=True, norm_type='l2', weights=[1.0, 1.0, 1.0])
# fused_vector = fusion.fuse([popmv_feat, dwt_feat, color_feat])
# print(fused_vector)
