import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
from typing import List, Tuple, Dict, Optional
import io
import base64

from core.dsu import DSU
from core.segmentation import ImageSegmentation

# Page configuration
st.set_page_config(
    page_title="Image Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'segmented_image' not in st.session_state:
        st.session_state.segmented_image = None
    if 'regions' not in st.session_state:
        st.session_state.regions = []
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {}

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Image Segmentation</h1>
        <p>Computer vision using DSU algorithms</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Image upload section
        st.subheader("📁 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image to analyze (max 10MB)"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            st.session_state.original_image = np.array(image)
            
            # Display image info
            st.success(f"Image loaded: {image.size[0]}×{image.size[1]} pixels")
        
        # Segmentation parameters
        st.subheader("🎨 Segmentation Parameters")
        
        color_threshold = st.slider(
            "Color Threshold",
            min_value=1,
            max_value=100,
            value=30,
            help="Lower values create more regions"
        )
        
        min_region_size = st.slider(
            "Min Region Size",
            min_value=1,
            max_value=1000,
            value=50,
            help="Filter out small regions"
        )
        
        connectivity = st.selectbox(
            "Pixel Connectivity",
            options=[4, 8],
            index=1,
            help="4-connected or 8-connected neighbors"
        )
        
        color_space = st.selectbox(
            "Color Space",
            options=['RGB', 'HSV', 'LAB'],
            index=0,
            help="Color space for similarity calculation"
        )
        
        # Process button
        if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
            if st.session_state.original_image is not None:
                process_image(
                    color_threshold,
                    min_region_size,
                    connectivity,
                    color_space
                )
            else:
                st.error("Please upload an image first!")
    
    # Main content area
    if st.session_state.original_image is not None:
        display_segmentation_comparison()
        display_export_options()
    else:
        # Welcome message
        st.markdown("""
        <div class="upload-section">
            <h3>👋 Welcome to Image Segmentation</h3>
            <p>Upload an image using the sidebar to get started with computer vision analysis.</p>
            <p><strong>Features:</strong></p>
            <ul style="text-align: left; display: inline-block;">
                <li>🎨 Intelligent image segmentation using DSU algorithms</li>
                <li>📊 Basic segmentation metrics</li>
                <li>💾 Export results as PNG</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def process_image(color_threshold, min_region_size, connectivity, color_space):
    """Process the image with given parameters"""
    with st.spinner("🔄 Processing image..."):
        start_time = time.time()
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Initialize segmentation
            status_text.text("Initializing segmentation...")
            progress_bar.progress(20)
            
            segmentation = ImageSegmentation(st.session_state.original_image)
            
            # Step 2: Perform segmentation
            status_text.text("Performing segmentation...")
            progress_bar.progress(60)
            
            regions = segmentation.segment(
                color_threshold=color_threshold,
                min_region_size=min_region_size,
                connectivity=connectivity,
                color_space=color_space
            )
            
            # Step 3: Generate segmented image
            status_text.text("Generating segmented image...")
            progress_bar.progress(90)
            
            segmented_image = segmentation.generate_segmented_image(regions)
            
            # Update session state
            st.session_state.regions = regions
            st.session_state.segmented_image = segmented_image
            
            # Calculate processing stats
            processing_time = time.time() - start_time
            st.session_state.processing_stats = {
                'total_pixels': st.session_state.original_image.shape[0] * st.session_state.original_image.shape[1],
                'total_regions': len(regions),
                'processing_time': processing_time,
                'parameters': {
                    'color_threshold': color_threshold,
                    'min_region_size': min_region_size,
                    'connectivity': connectivity,
                    'color_space': color_space
                }
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            st.success(f"Processing completed in {processing_time:.2f} seconds!")
            st.balloons()
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

def display_segmentation_comparison():
    """Display original vs segmented image comparison"""
    if st.session_state.segmented_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(st.session_state.original_image, use_column_width=True)
            
        with col2:
            st.subheader("🎨 Segmented Image")
            st.image(st.session_state.segmented_image, use_column_width=True)
        
        # Display basic stats
        if st.session_state.processing_stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Pixels", f"{st.session_state.processing_stats['total_pixels']:,}")
            
            with col2:
                st.metric("Regions Found", st.session_state.processing_stats['total_regions'])
            
            with col3:
                compression_ratio = (st.session_state.processing_stats['total_regions'] / 
                                   st.session_state.processing_stats['total_pixels']) * 100
                st.metric("Compression", f"{compression_ratio:.2f}%")
            
            with col4:
                st.metric("Processing Time", f"{st.session_state.processing_stats['processing_time']:.2f}s")
    
    else:
        st.info("Process an image to see the segmentation results.")

def display_export_options():
    """Display export options"""
    if st.session_state.segmented_image is not None:
        st.subheader("💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Image Export**")
            
            # Export segmented image
            if st.button("📷 Download Segmented Image"):
                img_buffer = io.BytesIO()
                Image.fromarray(st.session_state.segmented_image).save(img_buffer, format='PNG')
                st.download_button(
                    label="Download PNG",
                    data=img_buffer.getvalue(),
                    file_name=f"segmented_image_{int(time.time())}.png",
                    mime="image/png"
                )
        
        with col2:
            st.markdown("**Processing Report**")
            
            if st.button("📄 Generate Report"):
                if st.session_state.processing_stats:
                    report = generate_processing_report()
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"processing_report_{int(time.time())}.txt",
                        mime="text/plain"
                    )
    
    else:
        st.info("Process an image to access export options.")

def generate_processing_report():
    """Generate a basic processing report"""
    stats = st.session_state.processing_stats
    
    report = f"""
IMAGE SEGMENTATION REPORT
=========================

Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

IMAGE INFORMATION
-----------------
Total Pixels: {stats['total_pixels']:,}
Image Dimensions: {st.session_state.original_image.shape[1]} × {st.session_state.original_image.shape[0]}

SEGMENTATION RESULTS
--------------------
Total Regions: {stats['total_regions']}
Compression Ratio: {(stats['total_regions'] / stats['total_pixels']) * 100:.2f}%
Processing Time: {stats['processing_time']:.2f} seconds
Processing Speed: {stats['total_pixels']/stats['processing_time']:,.0f} pixels/second

PARAMETERS USED
---------------
Color Threshold: {stats['parameters']['color_threshold']}
Minimum Region Size: {stats['parameters']['min_region_size']} pixels
Pixel Connectivity: {stats['parameters']['connectivity']}-connected
Color Space: {stats['parameters']['color_space']}
"""
    
    return report

if __name__ == "__main__":
    main()
