import os
import numpy as np
import json
from typing import List, Dict, Optional
from src.preprocessing import ImagePreprocessor
from src.feature_extractors import POPMVExtractor, DWTEntropyExtractor, ColorStatsExtractor
from src.fusion import FeatureFusion
from utils.image_io import load_image

class FeatureIndexer:
    def __init__(self, 
                 config_path: str,
                 popmv_params: Optional[dict] = None,
                 dwt_params: Optional[dict] = None,
                 color_params: Optional[dict] = None,
                 fusion_params: Optional[dict] = None):
        """
        Initialize the feature indexer with configuration and extractor parameters.
        """
        self.preprocessor = ImagePreprocessor(config_path)
        self.popmv = POPMVExtractor(**(popmv_params or {}))
        self.dwt = DWTEntropyExtractor(**(dwt_params or {}))
        self.color = ColorStatsExtractor(**(color_params or {}))
        self.fusion = FeatureFusion(**(fusion_params or {}))

    def index_folder(self, image_folder: str, output_path: str) -> None:
        """
        Index all images in a folder and save the feature index to output_path (as .npy or .json).
        :param image_folder: Path to the folder containing images.
        :param output_path: Path to save the feature index (should end with .npy or .json).
        """
        index = []
        for fname in os.listdir(image_folder):
            fpath = os.path.join(image_folder, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                features = self._extract_features(fpath)
                index.append({
                    'image_id': fname,
                    'feature_vector': features.tolist()
                })
            except Exception as e:
                print(f"Failed to index {fname}: {e}")
        self._save_index(index, output_path)

    def _extract_features(self, image_path: str) -> np.ndarray:
        """
        Preprocess image, extract all features, and fuse them.
        :param image_path: Path to the image file.
        :return: Fused feature vector.
        """
        img = self.preprocessor.preprocess(image_path)
        # For color features, reload original (non-grayscale) image
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

    def _save_index(self, index: List[Dict], output_path: str) -> None:
        """
        Save the index to a file (supports .npy and .json).
        :param index: List of dicts with image_id and feature_vector.
        :param output_path: Path to save the index.
        """
        if output_path.endswith('.npy'):
            np.save(output_path, index)
        elif output_path.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(index, f)
        else:
            raise ValueError('Output path must end with .npy or .json')

# Example usage:
# indexer = FeatureIndexer(
#     config_path='../config.json',
#     popmv_params={'window_size': 3},
#     dwt_params={'wavelet': 'haar', 'level': 1},
#     color_params={'color_space': 'RGB'},
#     fusion_params={'normalize': True, 'norm_type': 'l2'}
# )
# indexer.index_folder('../data/images', '../data/features/features.json')
