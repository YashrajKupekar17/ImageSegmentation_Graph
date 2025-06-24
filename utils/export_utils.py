"""
Export utilities for saving results in various formats.
"""
# project/utils/export_utils.py

import numpy as np
import pandas as pd
import json
import cv2
from PIL import Image
from typing import List, Dict
import io

class ExportUtils:
    @staticmethod
    def save_image(image: np.ndarray, filename: str, format: str = 'PNG'):
        """
        Save image to file.
        
        Args:
            image: Image array to save
            filename: Output filename
            format: Image format (PNG, JPEG, etc.)
        """
        pil_image = Image.fromarray(image)
        pil_image.save(filename, format=format)
    
    @staticmethod
    def export_regions_csv(regions: List[Dict], filename: str):
        """
        Export region data to CSV file.
        
        Args:
            regions: List of region dictionaries
            filename: Output CSV filename
        """
        region_data = []
        for region in regions:
            region_data.append({
                'id': region['id'],
                'size': region['size'],
                'centroid_x': region['centroid'][0],
                'centroid_y': region['centroid'][1],
                'avg_color_r': region['avg_color'][0],
                'avg_color_g': region['avg_color'][1],
                'avg_color_b': region['avg_color'][2],
                'bbox_x': region['bounding_box']['x'],
                'bbox_y': region['bounding_box']['y'],
                'bbox_width': region['bounding_box']['width'],
                'bbox_height': region['bounding_box']['height']
            })
        
        df = pd.DataFrame(region_data)
        df.to_csv(filename, index=False)
    
    @staticmethod
    def export_mst_csv(mst_edges: List[Dict], filename: str):
        """
        Export MST edges to CSV file.
        
        Args:
            mst_edges: List of MST edge dictionaries
            filename: Output CSV filename
        """
        df = pd.DataFrame(mst_edges)
        df.to_csv(filename, index=False)
    
    @staticmethod
    def export_regions_json(regions: List[Dict], filename: str):
        """
        Export region data to JSON file.
        
        Args:
            regions: List of region dictionaries
            filename: Output JSON filename
        """
        # Convert numpy types to native Python types for JSON serialization
        serializable_regions = []
        for region in regions:
            serializable_region = {
                'id': int(region['id']),
                'size': int(region['size']),
                'centroid': [float(region['centroid'][0]), float(region['centroid'][1])],
                'avg_color': [int(region['avg_color'][0]), int(region['avg_color'][1]), int(region['avg_color'][2])],
                'bounding_box': {
                    'x': int(region['bounding_box']['x']),
                    'y': int(region['bounding_box']['y']),
                    'width': int(region['bounding_box']['width']),
                    'height': int(region['bounding_box']['height'])
                }
            }
            serializable_regions.append(serializable_region)
        
        with open(filename, 'w') as f:
            json.dump(serializable_regions, f, indent=2)
    
    @staticmethod
    def create_image_buffer(image: np.ndarray, format: str = 'PNG') -> io.BytesIO:
        """
        Create image buffer for download.
        
        Args:
            image: Image array
            format: Image format
            
        Returns:
            BytesIO buffer containing image data
        """
        buffer = io.BytesIO()
        pil_image = Image.fromarray(image)
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        return buffer