"""
Visualization utilities for image segmentation and MST analysis.
"""

# project/core/visualizer.py
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional

class Visualizer:
    def create_mst_visualization(self, segmented_image: np.ndarray, 
                                regions: List[Dict], mst_edges: List[Dict],
                                show_overlay: bool = True, show_labels: bool = False,
                                edge_thickness: int = 2) -> go.Figure:
        """
        Create interactive MST visualization using Plotly.
        
        Args:
            segmented_image: Segmented image array
            regions: List of region dictionaries
            mst_edges: List of MST edge dictionaries
            show_overlay: Whether to show MST overlay
            show_labels: Whether to show region labels
            edge_thickness: Thickness of MST edges
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add segmented image as background
        fig.add_trace(go.Image(z=segmented_image, name="Segmented Image"))
        
        if show_overlay and mst_edges:
            # Create region ID to region mapping
            region_map = {region['id']: region for region in regions}
            
            # Add MST edges
            edge_x = []
            edge_y = []
            
            for edge in mst_edges:
                from_region = region_map.get(edge['from'])
                to_region = region_map.get(edge['to'])
                
                if from_region and to_region:
                    edge_x.extend([from_region['centroid'][0], to_region['centroid'][0], None])
                    edge_y.extend([from_region['centroid'][1], to_region['centroid'][1], None])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(color='#ff6b35', width=edge_thickness),
                name='MST Edges',
                hoverinfo='skip'
            ))
            
            # Add region centroids
            centroid_x = [region['centroid'][0] for region in regions]
            centroid_y = [region['centroid'][1] for region in regions]
            centroid_text = [f"Region {region['id']}<br>Size: {region['size']} pixels" 
                           for region in regions]
            
            fig.add_trace(go.Scatter(
                x=centroid_x, y=centroid_y,
                mode='markers+text' if show_labels else 'markers',
                marker=dict(
                    color='#ff6b35',
                    size=8,
                    line=dict(color='white', width=2)
                ),
                text=[str(region['id']) for region in regions] if show_labels else None,
                textposition='top center',
                textfont=dict(color='white', size=10),
                name='Region Centroids',
                hovertext=centroid_text,
                hoverinfo='text'
            ))
        
        # Update layout
        fig.update_layout(
            title="MST Analysis Visualization",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='white',
            width=800,
            height=600
        )
        
        return fig
    
    def create_region_size_distribution(self, regions: List[Dict]) -> go.Figure:
        """
        Create region size distribution visualization.
        
        Args:
            regions: List of region dictionaries
            
        Returns:
            Plotly figure object
        """
        sizes = [region['size'] for region in regions]
        
        fig = px.histogram(
            x=sizes,
            nbins=20,
            title="Region Size Distribution",
            labels={'x': 'Region Size (pixels)', 'y': 'Frequency'},
            color_discrete_sequence=['#667eea']
        )
        
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_mst_weight_distribution(self, mst_edges: List[Dict]) -> go.Figure:
        """
        Create MST edge weight distribution visualization.
        
        Args:
            mst_edges: List of MST edge dictionaries
            
        Returns:
            Plotly figure object
        """
        weights = [edge['weight'] for edge in mst_edges]
        
        fig = px.histogram(
            x=weights,
            nbins=15,
            title="MST Edge Weight Distribution",
            labels={'x': 'Edge Weight', 'y': 'Frequency'},
            color_discrete_sequence=['#764ba2']
        )
        
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig