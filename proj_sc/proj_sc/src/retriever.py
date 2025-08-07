import os
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from src.preprocessing import ImagePreprocessor
from src.feature_extractors import POPMVExtractor, DWTEntropyExtractor, ColorStatsExtractor
from src.fusion import FeatureFusion
from utils.image_io import load_image

class ImageRetriever:
    def __init__(self,
                 config_path: str,
                 index_path: str,
                 popmv_params: Optional[dict] = None,
                 dwt_params: Optional[dict] = None,
                 color_params: Optional[dict] = None,
                 fusion_params: Optional[dict] = None,
                 distance_metric: str = 'euclidean'):
        """
        Initialize the image retriever with configuration, index, and extractor parameters.
        :param config_path: Path to preprocessing config file.
        :param index_path: Path to the feature index file (.json or .npy).
        :param popmv_params: Parameters for POPMVExtractor.
        :param dwt_params: Parameters for DWTEntropyExtractor.
        :param color_params: Parameters for ColorStatsExtractor.
        :param fusion_params: Parameters for FeatureFusion.
        :param distance_metric: Similarity metric ('euclidean' or 'cosine').
        """
        self.preprocessor = ImagePreprocessor(config_path)
        self.popmv = POPMVExtractor(**(popmv_params or {}))
        self.dwt = DWTEntropyExtractor(**(dwt_params or {}))
        self.color = ColorStatsExtractor(**(color_params or {}))
        self.fusion = FeatureFusion(**(fusion_params or {}))
        self.distance_metric = distance_metric
        self.index = self._load_index(index_path)

    def _load_index(self, index_path: str) -> List[Dict]:
        """
        Load the feature index from a .json or .npy file.
        :param index_path: Path to the index file.
        :return: List of dicts with 'image_id' and 'feature_vector'.
        """
        if index_path.endswith('.json'):
            with open(index_path, 'r') as f:
                return json.load(f)
        elif index_path.endswith('.npy'):
            return np.load(index_path, allow_pickle=True).tolist()
        else:
            raise ValueError('Index path must end with .json or .npy')

    def retrieve(self, query_image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve the top_k most similar images to the query image.
        :param query_image_path: Path to the query image.
        :param top_k: Number of top results to return.
        :return: List of tuples (image_id, similarity_score), sorted by similarity.
        """
        query_feat = self._extract_query_features(query_image_path)
        results = []
        for entry in self.index:
            db_feat = np.array(entry['feature_vector'], dtype=np.float32)
            score = self._compute_distance(query_feat, db_feat)
            results.append((entry['image_id'], score))
        # For Euclidean, lower is better; for cosine, higher is better (so sort accordingly)
        reverse = self.distance_metric == 'cosine'
        results.sort(key=lambda x: x[1], reverse=reverse)
        return results[:top_k]

    def _extract_query_features(self, image_path: str) -> np.ndarray:
        """
        Preprocess and extract fused features from the query image.
        :param image_path: Path to the query image.
        :return: Fused feature vector.
        """
        img = self.preprocessor.preprocess(image_path)
        color_img = load_image(image_path, color=True)
        # Ensure grayscale for POPMV and DWT
        if len(img.shape) == 3:
            import cv2
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        popmv_feat = self.popmv.extract(gray_img)
        dwt_feat = self.dwt.extract(gray_img)
        color_feat = self.color.extract(color_img)
        fused = self.fusion.fuse([popmv_feat, dwt_feat, color_feat])
        return fused

    def _compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute similarity/distance between two feature vectors.
        :param vec1: Query feature vector.
        :param vec2: Database feature vector.
        :return: Similarity score (lower is better for Euclidean, higher is better for cosine).
        """
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(vec1 - vec2)
        elif self.distance_metric == 'cosine':
            num = np.dot(vec1, vec2)
            denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            return num / denom if denom > 0 else 0.0
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

# Example usage:
# retriever = ImageRetriever(
#     config_path='../config.json',
#     index_path='../data/features/features.json',
#     popmv_params={'window_size': 3},
#     dwt_params={'wavelet': 'haar', 'level': 1},
#     color_params={'color_space': 'RGB'},
#     fusion_params={'normalize': True, 'norm_type': 'l2'},
#     distance_metric='euclidean'
# )
# results = retriever.retrieve('../data/query/query1.jpg', top_k=5)
# print(results)
