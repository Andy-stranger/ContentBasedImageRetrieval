import numpy as np
import pywt
from typing import List, Tuple

class DWTEntropyExtractor:
    def __init__(self, wavelet: str = 'haar', level: int = 1):
        """
        Initialize the DWT Entropy extractor.
        :param wavelet: Wavelet type (e.g., 'haar', 'db1', 'db2').
        :param level: Number of decomposition levels.
        """
        self.wavelet = wavelet
        self.level = level

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract DWT entropy feature vector from a grayscale image.
        :param image: Preprocessed grayscale image as a NumPy array.
        :return: Feature vector (entropy values from sub-bands).
        """
        coeffs = self._dwt_decompose(image)
        entropies = self._compute_entropies(coeffs)
        return np.array(entropies, dtype=np.float32)

    def _dwt_decompose(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Perform multi-level DWT decomposition.
        :param image: Grayscale image.
        :return: List of sub-band coefficient arrays.
        """
        coeffs = pywt.wavedec2(image, self.wavelet, level=self.level)
        subbands = []
        # coeffs[0] is the approximation (LL), coeffs[1:] are details (LH, HL, HH)
        subbands.append(coeffs[0])
        for details in coeffs[1:]:
            subbands.extend(details)  # details is a tuple (LH, HL, HH)
        return subbands

    def _compute_entropies(self, subbands: List[np.ndarray]) -> List[float]:
        """
        Compute entropy for each sub-band.
        :param subbands: List of sub-band arrays.
        :return: List of entropy values.
        """
        return [self._entropy(sb) for sb in subbands]

    def _entropy(self, arr: np.ndarray) -> float:
        """
        Calculate the entropy of a sub-band.
        :param arr: Sub-band array.
        :return: Entropy value.
        """
        hist, _ = np.histogram(arr.flatten(), bins=256, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist)) if hist.size > 0 else 0.0
