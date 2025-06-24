"""
Distance metrics for color similarity calculation in different color spaces.
"""
# project/utils/distance_metrics.py

import numpy as np
import math
from typing import Dict

class DistanceMetrics:
    @staticmethod
    def calculate_distance(pixel1: Dict, pixel2: Dict, color_space: str = 'RGB') -> float:
        """
        Calculate color distance between two pixels in specified color space.
        
        Args:
            pixel1: First pixel dictionary with r, g, b values
            pixel2: Second pixel dictionary with r, g, b values
            color_space: Color space for distance calculation
            
        Returns:
            Color distance as float
        """
        if color_space == 'RGB':
            return DistanceMetrics._euclidean_distance_rgb(pixel1, pixel2)
        elif color_space == 'HSV':
            return DistanceMetrics._hsv_distance(pixel1, pixel2)
        elif color_space == 'LAB':
            return DistanceMetrics._lab_distance(pixel1, pixel2)
        else:
            return DistanceMetrics._euclidean_distance_rgb(pixel1, pixel2)
    
    @staticmethod
    def _euclidean_distance_rgb(pixel1: Dict, pixel2: Dict) -> float:
        """
        Calculate Euclidean distance in RGB color space.
        
        Args:
            pixel1: First pixel dictionary
            pixel2: Second pixel dictionary
            
        Returns:
            RGB distance
        """
        dr = pixel1['r'] - pixel2['r']
        dg = pixel1['g'] - pixel2['g']
        db = pixel1['b'] - pixel2['b']
        return math.sqrt(dr * dr + dg * dg + db * db)
    
    @staticmethod
    def _rgb_to_hsv(r: int, g: int, b: int) -> tuple:
        """
        Convert RGB to HSV color space.
        
        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            
        Returns:
            Tuple of (h, s, v) values
        """
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        delta = max_val - min_val
        
        # Hue calculation
        if delta == 0:
            h = 0
        elif max_val == r:
            h = ((g - b) / delta) % 6
        elif max_val == g:
            h = (b - r) / delta + 2
        else:
            h = (r - g) / delta + 4
        
        h = (h * 60 + 360) % 360
        
        # Saturation calculation
        s = 0 if max_val == 0 else delta / max_val
        
        # Value calculation
        v = max_val
        
        return h, s * 100, v * 100
    
    @staticmethod
    def _hsv_distance(pixel1: Dict, pixel2: Dict) -> float:
        """
        Calculate distance in HSV color space.
        
        Args:
            pixel1: First pixel dictionary
            pixel2: Second pixel dictionary
            
        Returns:
            HSV distance
        """
        h1, s1, v1 = DistanceMetrics._rgb_to_hsv(pixel1['r'], pixel1['g'], pixel1['b'])
        h2, s2, v2 = DistanceMetrics._rgb_to_hsv(pixel2['r'], pixel2['g'], pixel2['b'])
        
        # Handle hue circularity
        h_diff = abs(h1 - h2)
        h_diff = min(h_diff, 360 - h_diff)
        
        # Normalize and calculate distance
        h_norm = h_diff / 180  # Normalize to [0, 2]
        s_norm = (s1 - s2) / 100
        v_norm = (v1 - v2) / 100
        
        return math.sqrt(h_norm * h_norm + s_norm * s_norm + v_norm * v_norm)
    
    @staticmethod
    def _rgb_to_lab(r: int, g: int, b: int) -> tuple:
        """
        Convert RGB to LAB color space.
        
        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            
        Returns:
            Tuple of (l, a, b) values
        """
        # Normalize RGB values
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        
        # Gamma correction
        r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
        g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
        b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92
        
        # Convert to XYZ using sRGB matrix
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        
        # Normalize for D65 illuminant
        x /= 0.95047
        y /= 1.00000
        z /= 1.08883
        
        # XYZ to LAB
        x = x ** (1/3) if x > 0.008856 else (7.787 * x + 16/116)
        y = y ** (1/3) if y > 0.008856 else (7.787 * y + 16/116)
        z = z ** (1/3) if z > 0.008856 else (7.787 * z + 16/116)
        
        l = 116 * y - 16
        a = 500 * (x - y)
        b_lab = 200 * (y - z)
        
        return l, a, b_lab
    
    @staticmethod
    def _lab_distance(pixel1: Dict, pixel2: Dict) -> float:
        """
        Calculate distance in LAB color space.
        
        Args:
            pixel1: First pixel dictionary
            pixel2: Second pixel dictionary
            
        Returns:
            LAB distance
        """
        l1, a1, b1 = DistanceMetrics._rgb_to_lab(pixel1['r'], pixel1['g'], pixel1['b'])
        l2, a2, b2 = DistanceMetrics._rgb_to_lab(pixel2['r'], pixel2['g'], pixel2['b'])
        
        dl = l1 - l2
        da = a1 - a2
        db = b1 - b2
        
        return math.sqrt(dl * dl + da * da + db * db)