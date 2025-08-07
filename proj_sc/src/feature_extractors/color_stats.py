import numpy as np
import cv2
from scipy.stats import skew
from typing import Literal

class ColorStatsExtractor:
    def __init__(self, color_space: Literal['RGB', 'HSV'] = 'RGB'):
        """
        Initialize the color statistics extractor.
        :param color_space: 'RGB' or 'HSV'.
        """
        self.color_space = color_space.upper()
        if self.color_space not in ['RGB', 'HSV']:
            raise ValueError("color_space must be 'RGB' or 'HSV'")

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract mean, std, and skewness for each channel in the selected color space.
        :param image: Preprocessed color image as a NumPy array (BGR by OpenCV default).
        :return: Feature vector (mean, std, skewness for each channel).
        """
        img = self._convert_color_space(image)
        features = []
        for ch in range(img.shape[2]):
            channel = img[:, :, ch].flatten()
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(skew(channel))
        return np.array(features, dtype=np.float32)

    def _convert_color_space(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to the selected color space.
        :param image: Input image (BGR by OpenCV default).
        :return: Image in the selected color space.
        """
        if self.color_space == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.color_space == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError(f"Unsupported color space: {self.color_space}")
