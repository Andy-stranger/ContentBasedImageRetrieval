import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def load_image(path: str, color: bool = True) -> np.ndarray:
    """
    Load an image from disk.
    :param path: Path to the image file.
    :param color: If True, load as color (BGR). If False, load as grayscale.
    :return: Image as a NumPy array.
    """
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img

def save_image(path: str, image: np.ndarray) -> None:
    """
    Save an image to disk.
    :param path: Path to save the image.
    :param image: Image as a NumPy array.
    """
    cv2.imwrite(path, image)

def display_image(image: np.ndarray, title: Optional[str] = None, cmap: Optional[str] = None) -> None:
    """
    Display an image using matplotlib.
    :param image: Image as a NumPy array.
    :param title: Optional title for the plot.
    :param cmap: Optional colormap (e.g., 'gray' for grayscale images).
    """
    plt.figure()
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        plt.imshow(image, cmap=cmap or 'gray')
    else:
        # Convert BGR (OpenCV default) to RGB for display
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
