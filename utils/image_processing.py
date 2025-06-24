"""
Image processing utilities for preprocessing and manipulation.
"""
# project/utils/image_processing.py

import numpy as np
import cv2
from PIL import Image
from typing import Tuple

class ImageProcessor:
    @staticmethod
    def resize_image(image: np.ndarray, max_size: int) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image array
            max_size: Maximum dimension size
            
        Returns:
            Resized image array
        """
        height, width = image.shape[:2]
        
        if max(height, width) <= max_size:
            return image
        
        if height > width:
            new_height = max_size
            new_width = int(width * max_size / height)
        else:
            new_width = max_size
            new_height = int(height * max_size / width)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply Gaussian blur to image.
        
        Args:
            image: Input image array
            sigma: Blur strength (standard deviation)
            
        Returns:
            Blurred image array
        """
        # Calculate kernel size based on sigma
        kernel_size = int(2 * np.ceil(2 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def convert_color_space(image: np.ndarray, color_space: str) -> np.ndarray:
        """
        Convert image to different color space.
        
        Args:
            image: Input RGB image array
            color_space: Target color space ('RGB', 'HSV', 'LAB')
            
        Returns:
            Converted image array
        """
        if color_space == 'RGB':
            return image
        elif color_space == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LAB':
            return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        else:
            return image
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image array
        """
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """
        Denormalize image from [0, 1] to [0, 255] range.
        
        Args:
            image: Normalized image array
            
        Returns:
            Denormalized image array
        """
        return (image * 255).astype(np.uint8)