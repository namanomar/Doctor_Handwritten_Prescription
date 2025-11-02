---


---
# 🩺 Doctor Handwritten Prescription Classifier

A deep learning-based web application for classifying handwritten prescription images into medicine names using VGG16 transfer learning and advanced image preprocessing techniques.

![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Image Processing Pipeline](#image-processing-pipeline)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## ✨ Features

- **Deep Learning Classification**: VGG16-based model trained on handwritten prescription dataset
- **Advanced Image Preprocessing**: Multi-step image analysis pipeline including:
  - Noise Reduction (Gaussian, Median, Bilateral)
  - Image Enhancement (Brightness & Contrast adjustment)
  - Histogram Analysis with statistical insights
  - Binary Thresholding (Otsu, Adaptive, Simple)
  - Morphological Operations (Opening, Closing, Erosion, Dilation)
  - Image Segmentation (Watershed, Contour detection)
- **Interactive Web UI**: Modern, responsive interface with step-by-step visualization
- **RESTful API**: Comprehensive API endpoints for individual processing steps
- **Real-time Analysis**: Step-by-step visualization of preprocessing pipeline

## 🏗️ Architecture

### Model Architecture

- **Base Model**: VGG16 (Transfer Learning)
- **Input Size**: 128x128 RGB images
- **Output**: Multi-class classification (78 medicine categories)
- **Framework**: TensorFlow/Keras

### Image Processing Pipeline

```
Original Image 
  ↓
Noise Reduction (Gaussian Blur)
  ↓
Image Enhancement (Brightness & Contrast)
  ↓
Histogram Analysis
  ↓
Binary Thresholding (Otsu)
  ↓
Morphological Operations (Opening)
  ↓
Model Prediction
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Local Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Doctor_Handwritten_Prescription
   ```
2. **Create virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Ensure model files are in place**

   ```
   model/
   ├── model_vgg16.h5
   └── class_names.json
   ```

## 💻 Usage

### Running the Application

1. **Start the FastAPI server**

   ```bash
   cd api
   python main.py
   ```

   Or using uvicorn directly:

   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
2. **Access the application**

   - Web UI: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc

### Using the Web Interface

1. Open http://localhost:8000 in your browser
2. Upload a handwritten prescription image (PNG, JPG, JPEG)
3. Click "🔍 Predict Medicine" button
4. View the step-by-step preprocessing pipeline and final predictions

## 🔌 API Endpoints

### Main Endpoints

#### `POST /process_and_predict`

Complete pipeline: Process image through all steps then predict

- **Request**: Multipart form data with `file` (image)
- **Response**: JSON with processing steps and predictions

#### `POST /predict`

Simple prediction without processing steps

- **Request**: Multipart form data with `file` (image)
- **Response**: JSON with predictions and preview

### Individual Processing Endpoints

#### `POST /noise_reduction`

Apply noise reduction to image

- **Parameters**:
  - `file`: Image file (required)
  - `method`: "gaussian" | "median" | "bilateral" (default: "gaussian")
  - `kernel_size`: int (default: 5)

#### `POST /enhancement`

Enhance image brightness and contrast

- **Parameters**:
  - `file`: Image file (required)
  - `brightness`: float (default: 20.0, range: -255 to 255)
  - `contrast`: float (default: 1.2, range: 0.0 to 3.0)

#### `POST /binary_threshold`

Apply binary thresholding

- **Parameters**:
  - `file`: Image file (required)
  - `method`: "otsu" | "adaptive" | "simple" (default: "otsu")
  - `threshold_value`: int (default: 127, only for "simple")

#### `POST /morphological`

Apply morphological operations

- **Parameters**:
  - `file`: Image file (required)
  - `operation`: "opening" | "closing" | "erosion" | "dilation" | "gradient" (default: "opening")
  - `kernel_size`: int (default: 5)
  - `iterations`: int (default: 1)

#### `POST /segmentation`

Apply image segmentation

- **Parameters**:
  - `file`: Image file (required)
  - `method`: "watershed" | "contour" (default: "watershed")

#### `POST /histogram_analysis`

Perform histogram analysis

- **Parameters**:
  - `file`: Image file (required)
- **Response**: Histogram image and statistics (mean, std, min, max)

### Example API Usage

```bash
# Using curl
curl -X POST "http://localhost:8000/noise_reduction" \
  -F "file=@prescription.png" \
  -F "method=gaussian" \
  -F "kernel_size=5"

# Using Python requests
import requests

files = {'file': open('prescription.png', 'rb')}
data = {'method': 'gaussian', 'kernel_size': 5}
response = requests.post('http://localhost:8000/noise_reduction', 
                         files=files, data=data)
print(response.json())
```

## 📊 Image Processing Pipeline

### Step 1: Original Image

- Loads and displays the input image

### Step 2: Noise Reduction

- Applies Gaussian blur to reduce noise and artifacts
- Improves image quality for further processing

### Step 3: Image Enhancement

- Adjusts brightness (+20)
- Enhances contrast (1.2x)
- Improves visibility of handwritten text

### Step 4: Histogram Analysis

- Calculates pixel intensity distribution
- Provides statistics: mean, standard deviation, min, max values
- Visualizes histogram as a chart

### Step 5: Binary Thresholding

- Converts image to binary (black and white)
- Uses Otsu's method for automatic threshold selection
- Separates text from background

### Step 6: Morphological Analysis

- Applies opening operation to remove small noise
- Refines text boundaries
- Cleans up the binary image

### Step 7: Image Segmentation

- Identifies and segments different regions
- Uses watershed algorithm for region detection
- Marks boundaries for visualization

### Step 8: Model Prediction

- Preprocesses final image for VGG16 model
- Generates predictions for 78 medicine categories
- Returns top 5 predictions with confidence scores

## 🐳 Deployment

### Using Docker

1. **Build the Docker image**

   ```bash
   docker build -t prescription-classifier .
   ```
2. **Run the container**

   ```bash
   docker run -p 8000:8000 prescription-classifier
   ```

### Deploying to Hugging Face Spaces

1. **Prepare your repository**

   - Ensure all files are committed
   - Include `requirements.txt`, `Dockerfile`, and `README.md`
2. **Push to Hugging Face**

   ```bash
   git push origin main
   ```
3. **Create Space on Hugging Face**

   - Go to https://huggingface.co/spaces
   - Create new Space
   - Select "Docker" as SDK
   - Connect your repository
4. **Configure Space Settings**

   - Hardware: CPU (or GPU if needed)
   - Environment variables: Set if needed

### Environment Variables

No environment variables are required by default. Model paths are relative to the application directory.

## 📁 Project Structure

```
Doctor_Handwritten_Prescription/
├── api/
│   ├── main.py              # FastAPI application and endpoints
│   └── image_processing.py  # Image preprocessing functions
├── model/
│   ├── model_vgg16.h5       # Trained VGG16 model
│   └── class_names.json     # Medicine class mappings
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
└── README.md               # This file
```

## 🔧 Configuration

### Model Configuration

- **Model Path**: `model/model_vgg16.h5`
- **Classes Path**: `model/class_names.json`
- **Image Size**: 128x128 pixels
- **Top K Predictions**: 5

### Default Processing Parameters

- **Noise Reduction**: Gaussian, kernel_size=5
- **Enhancement**: brightness=20, contrast=1.2
- **Thresholding**: Otsu method
- **Morphological**: Opening, kernel_size=5, iterations=1

## 📝 Model Information

- **Architecture**: VGG16 (Transfer Learning)
- **Input Shape**: (128, 128, 3)
- **Output Classes**: 78 medicine categories
- **Training**: Fine-tuned on Doctor's Handwritten Prescription BD dataset

### Supported Medicine Classes (78 categories)

Examples: Ace, Aceta, Alatrol, Amodis, Atrizin, Azithrocin, Napa, Nexum, Omastin, and 70+ more...

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is for academic purposes (CSE 411 Computer Vision Principles and Process).

## 🙏 Acknowledgments

- Doctor's Handwritten Prescription BD Dataset
- TensorFlow and Keras teams
- FastAPI framework
- OpenCV and scikit-image communities

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Note**: This project is developed for academic purposes as part of CSE 411 Computer Vision Principles and Process course.
