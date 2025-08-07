import numpy as np
from typing import Tuple

class POPMVExtractor:
    def __init__(self, window_size: int = 3):
        """
        Initialize the POPMV extractor.
        :param window_size: Size of the neighborhood window (default 3x3).
        """
        self.window_size = window_size
        if self.window_size % 2 == 0 or self.window_size < 3:
            raise ValueError("window_size must be an odd integer >= 3")

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the POPMV feature vector from a grayscale image.
        :param image: Preprocessed grayscale image as a NumPy array.
        :return: POPMV feature vector as a NumPy array.
        """
        poop_codes = self._compute_poopp(image)
        feature_vector = self._majority_voting(poop_codes)
        return feature_vector

    def _compute_poopp(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the Peak Oriented Octal Pattern (POOP) for each pixel.
        :param image: Grayscale image.
        :return: POOP code image (same size as input, dtype=np.uint8).
        """
        # Pad image to handle borders
        pad = self.window_size // 2
        padded = np.pad(image, pad, mode='reflect')
        poop_codes = np.zeros_like(image, dtype=np.uint8)
        # Define 8-neighborhood offsets (clockwise)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                   (1, 1), (1, 0), (1, -1), (0, -1)]
        for i in range(pad, pad + image.shape[0]):
            for j in range(pad, pad + image.shape[1]):
                center = padded[i, j]
                neighbors = [padded[i + dx, j + dy] for dx, dy in offsets]
                # Find peak orientation (max diff)
                diffs = [abs(int(center) - int(n)) for n in neighbors]
                peak_idx = int(np.argmax(diffs))
                # Octal pattern: 1 if neighbor >= center, else 0
                pattern = [(1 if neighbors[k] >= center else 0) for k in range(8)]
                # Rotate pattern to align with peak orientation
                rotated = pattern[peak_idx:] + pattern[:peak_idx]
                # Convert to integer (octal code)
                code = sum([bit << (7 - idx) for idx, bit in enumerate(rotated)])
                poop_codes[i - pad, j - pad] = code
        return poop_codes

    def _majority_voting(self, poop_codes: np.ndarray) -> np.ndarray:
        """
        Apply majority voting to POOP codes to get the final POPMV feature vector.
        :param poop_codes: POOP code image.
        :return: Feature vector (histogram of codes, normalized).
        """
        # There are 256 possible 8-bit codes
        hist, _ = np.histogram(poop_codes, bins=256, range=(0, 256))
        # Majority voting: keep only the most frequent codes (e.g., top 16)
        top_k = 16
        top_indices = np.argsort(hist)[-top_k:][::-1]
        feature_vector = hist[top_indices].astype(np.float32)
        # Normalize feature vector
        if feature_vector.sum() > 0:
            feature_vector /= feature_vector.sum()
        return feature_vector
