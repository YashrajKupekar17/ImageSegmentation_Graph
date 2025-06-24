# 🧠 Image Segmentation App

A streamlined image processing application that performs intelligent image segmentation using **Disjoint Set Union (DSU)** algorithms with Python and Streamlit. This app focuses on efficient pixel clustering for clean, accurate image segmentation.

## 🚀 Features

### 🎨 Smart Image Segmentation
- **DSU-based pixel clustering** with path compression and union by rank optimization
- **Multiple connectivity patterns** (4-connected or 8-connected neighbors)
- **Advanced color spaces** (RGB, HSV, LAB) for better perceptual accuracy
- **Configurable similarity thresholds** for fine-tuned segmentation control
- **Region filtering** by minimum size to remove noise
- **Distinct color visualization** for clear region identification

### 📊 Essential Analytics
- **Real-time processing statistics** with performance metrics
- **Basic region analysis** (count, size distribution, processing time)
- **Compression ratio** showing segmentation efficiency
- **Clean side-by-side comparison** of original vs segmented images

### 💾 Simple Export
- **PNG image download** for segmented results
- **One-click export** with timestamped filenames
- **High-quality output** preserving segmentation details


## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone 
   cd image-segmentation-app
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run main.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 🎯 Usage Guide

### 1. **Image Upload**
- Use the sidebar file uploader to select an image (PNG, JPG, JPEG, BMP, TIFF)
- Supported formats with automatic format detection
- Instant image information display upon upload

### 2. **Segmentation Parameters**
- **Color Threshold**: Control region merging sensitivity (1-100)
  - Lower values create more detailed segments
  - Higher values merge similar regions more aggressively
- **Min Region Size**: Filter out small noise regions (1-1000 pixels)
  - Removes tiny segments that may be noise
  - Keeps only meaningful regions
- **Connectivity**: Choose 4-connected or 8-connected neighbors
  - 4-connected: Only horizontal/vertical neighbors
  - 8-connected: Includes diagonal neighbors
- **Color Space**: Select RGB, HSV, or LAB for distance calculations
  - RGB: Standard color representation
  - HSV: Better for color-based segmentation
  - LAB: Perceptually uniform color space

### 3. **Processing & Results**
- Click "🚀 Analyze Image" to start segmentation
- View real-time processing feedback
- Compare original and segmented images side-by-side
- Review basic statistics and metrics

### 4. **Export Results**
- Download segmented image as PNG with one click
- Timestamped filenames for easy organization
- High-quality output preserving all segmentation details

## 🔬 Algorithm Details

### Disjoint Set Union (DSU)
- **Path Compression**: Optimizes find operations to O(α(n)) amortized time
- **Union by Rank**: Maintains balanced tree structure for efficient unions
- **Applications**: Efficiently groups similar pixels into connected regions
- **Memory Efficient**: Minimal overhead for large images

### Color Distance Calculation
- **Euclidean Distance**: Standard distance metric in chosen color space
- **Threshold-based Merging**: Pixels within threshold distance are grouped
- **Neighbor Analysis**: Considers spatial connectivity for region formation

### Distinct Color Generation
- **HSV Color Space**: Generates visually distinct colors for regions
- **Golden Angle Distribution**: Ensures maximum color separation
- **Automatic Scaling**: Adapts to any number of regions

## 📈 Performance Features

- **Efficient Processing**: Optimized algorithms for real-time interaction
- **Memory Management**: Handles large images without memory issues
- **Progress Feedback**: Real-time processing status updates
- **Error Handling**: Robust error management with user-friendly messages

## 🎨 Key Benefits

### Simplicity
- **Clean Interface**: Focused on essential segmentation features
- **Easy Parameters**: Intuitive controls with helpful tooltips
- **Quick Results**: Fast processing with immediate visual feedback

### Quality
- **Accurate Segmentation**: DSU ensures mathematically sound region formation
- **Visual Clarity**: Distinct colors make regions easy to identify
- **Flexible Parameters**: Fine-tune results for different image types

### Efficiency
- **Minimal Dependencies**: Lightweight installation and deployment
- **Fast Processing**: Optimized algorithms for quick results
- **Low Memory**: Efficient data structures minimize resource usage

## 🛠️ Technical Requirements

### Dependencies
```txt
streamlit>=1.28.0
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
matplotlib>=3.7.0
plotly>=5.15.0
scikit-image>=0.21.0
networkx>=3.1
pandas>=2.0.0
scipy>=1.11.0
opencv-python
opencv-python-headless
opencv-contrib-python
```

---

**Built with ❤️ using Python, Streamlit, and efficient computer vision algorithms.**

*Simple. Fast. Effective.*
