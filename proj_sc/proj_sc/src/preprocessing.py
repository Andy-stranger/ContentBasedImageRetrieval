import os
import json
import numpy as np
import cv2
from typing import Tuple, Optional
from utils.image_io import load_image, save_image

class ImagePreprocessor:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.settings = self.config.get('preprocessing', {})

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return json.load(f)

    def preprocess(self, image_path: str) -> np.ndarray:
        image = self._load_image(image_path)
        if self.settings.get('resize', False):
            image = self._resize(image, tuple(self.settings.get('resize_shape', [256, 256])))
        if self.settings.get('to_grayscale', False):
            image = self._to_grayscale(image)
        if self.settings.get('normalize', False):
            image = self._normalize(image)
        if self.settings.get('noise_reduction', False):
            image = self._reduce_noise(image)
        return image

    def _load_image(self, image_path: str) -> np.ndarray:
        return load_image(image_path, color=True)

    def _resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image  # Already grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        return image

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        # Use Gaussian blur for noise reduction
        return cv2.GaussianBlur(image, (3, 3), 0)

    def batch_preprocess(self, input_dir: str, output_dir: Optional[str] = None) -> None:
        os.makedirs(output_dir, exist_ok=True) if output_dir else None
        for fname in os.listdir(input_dir):
            fpath = os.path.join(input_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                processed = self.preprocess(fpath)
                if output_dir:
                    out_path = os.path.join(output_dir, fname)
                    self._save_image(processed, out_path)
            except Exception as e:
                print(f"Failed to process {fname}: {e}")

    def _save_image(self, image: np.ndarray, path: str) -> None:
        # If normalized, convert back to 0-255 for saving
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        save_image(path, image)

# Example usage (uncomment for testing):
# preprocessor = ImagePreprocessor(config_path='../config.json')
# preprocessed_img = preprocessor.preprocess('../data/images/sample.jpg')
# preprocessor.batch_preprocess('../data/images', '../data/processed')
