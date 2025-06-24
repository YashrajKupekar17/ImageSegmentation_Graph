import numpy as np
import time
from typing import List, Dict, Tuple
from .dsu import DSU

class ImageSegmentation:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.pixels = self._extract_pixels()
        self.dsu = DSU(len(self.pixels))
    
    def _extract_pixels(self) -> List[Dict]:
        pixels = []
        for y in range(self.height):
            for x in range(self.width):
                pixel = {
                    'x': x,
                    'y': y,
                    'r': int(self.image[y, x, 0]),
                    'g': int(self.image[y, x, 1]),
                    'b': int(self.image[y, x, 2])
                }
                pixels.append(pixel)
        return pixels
    
    def _get_neighbors(self, x: int, y: int, connectivity: int) -> List[int]:
        neighbors = []
        
        if connectivity == 4:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                      (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append(ny * self.width + nx)
        
        return neighbors
    
    def _color_distance(self, pixel1: Dict, pixel2: Dict, color_space: str = 'RGB') -> float:
        if color_space == 'RGB':
            dr = pixel1['r'] - pixel2['r']
            dg = pixel1['g'] - pixel2['g']
            db = pixel1['b'] - pixel2['b']
            return (dr * dr + dg * dg + db * db) ** 0.5
        # Add other color spaces if needed
        return 0.0
    
    def segment(self, color_threshold: float = 30, min_region_size: int = 50,
                connectivity: int = 8, color_space: str = 'RGB') -> List[Dict]:
        start_time = time.time()
        
        self.dsu = DSU(len(self.pixels))
        
        for i, pixel in enumerate(self.pixels):
            neighbors = self._get_neighbors(pixel['x'], pixel['y'], connectivity)
            
            for neighbor_idx in neighbors:
                neighbor = self.pixels[neighbor_idx]
                distance = self._color_distance(pixel, neighbor, color_space)
                
                if distance <= color_threshold:
                    self.dsu.union(i, neighbor_idx)
        
        # Build regions
        component_map = {}
        for i in range(len(self.pixels)):
            root = self.dsu.find(i)
            if root not in component_map:
                component_map[root] = []
            component_map[root].append(i)
        
        regions = []
        region_id = 0
        
        for root, pixel_indices in component_map.items():
            if len(pixel_indices) >= min_region_size:
                region_pixels = [self.pixels[idx] for idx in pixel_indices]
                region = self._create_region(region_id, region_pixels)
                regions.append(region)
                region_id += 1
        
        return regions
    
    def _create_region(self, region_id: int, pixels: List[Dict]) -> Dict:
        sum_x = sum(p['x'] for p in pixels)
        sum_y = sum(p['y'] for p in pixels)
        centroid = (sum_x / len(pixels), sum_y / len(pixels))
        
        sum_r = sum(p['r'] for p in pixels)
        sum_g = sum(p['g'] for p in pixels)
        sum_b = sum(p['b'] for p in pixels)
        avg_color = (
            int(sum_r / len(pixels)),
            int(sum_g / len(pixels)),
            int(sum_b / len(pixels))
        )
        
        return {
            'id': region_id,
            'pixels': pixels,
            'centroid': centroid,
            'avg_color': avg_color,
            'size': len(pixels)
        }
    
    def generate_segmented_image(self, regions: List[Dict]) -> np.ndarray:
        segmented = np.zeros_like(self.image)
        colors = self._generate_distinct_colors(len(regions))
        
        pixel_to_region = {}
        for i, region in enumerate(regions):
            for pixel in region['pixels']:
                key = (pixel['x'], pixel['y'])
                pixel_to_region[key] = colors[i]
        
        for y in range(self.height):
            for x in range(self.width):
                key = (x, y)
                if key in pixel_to_region:
                    segmented[y, x] = pixel_to_region[key]
                else:
                    segmented[y, x] = [128, 128, 128]
        
        return segmented

    def _generate_distinct_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(num_colors):
            hue = (i * 137.508) % 360
            saturation = 70 + (i % 3) * 15
            value = 80 + (i % 2) * 20
            
            h, s, v = hue / 360, saturation / 100, value / 100
            rgb = self._hsv_to_rgb(h, s, v)
            colors.append(tuple(int(c * 255) for c in rgb))
        
        return colors

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if i % 6 == 0: return v, t, p
        elif i % 6 == 1: return q, v, p
        elif i % 6 == 2: return p, v, t
        elif i % 6 == 3: return p, q, v
        elif i % 6 == 4: return t, p, v
        else: return v, p, q
