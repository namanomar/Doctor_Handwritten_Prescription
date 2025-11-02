from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import json
import os
import tensorflow as tf
import base64
import uvicorn
from api.image_processing import (
    process_image_pipeline, 
    load_image_from_bytes,
    noise_reduction,
    enhance_brightness_contrast,
    binary_thresholding,
    morphological_operations,
    image_segmentation,
    histogram_analysis,
    histogram_to_base64,
    image_to_base64
)

app = FastAPI(title="Handwritten Prescription Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model_vgg16.h5")           
CLASSES_PATH = os.path.join(PROJECT_ROOT, "model", "class_names.json")      
IMAGE_SIZE = (128, 128)             
SCALE_0_1 = True                     
TOP_K = 5                            

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
else:
    num_classes = model.output_shape[-1] if isinstance(model.output_shape, tuple) else model.output_shape[-1]
    idx_to_class = {str(i): str(i) for i in range(num_classes)}

num_classes = len(idx_to_class) if isinstance(idx_to_class, dict) else model.output_shape[-1]


def decode_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)   # ensure RGB
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return img, tf.one_hot(label, depth=num_classes)

def preprocess_image_bytes(image_bytes: bytes):
    img_tensor = tf.constant(image_bytes)
    img = tf.image.decode_image(img_tensor, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.0
    # Convert to numpy and add batch dimension
    img_np = np.expand_dims(img.numpy(), axis=0)
    return img_np.astype("float32")


@app.get("/", response_class=HTMLResponse)
async def index():
    html = """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Doctor Handwritten Prescription Classifier</title>
        <style>
          * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }
          
          body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
          }
          
          .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
            max-width: 800px;
            width: 100%;
            animation: slideUp 0.5s ease-out;
          }
          
          @keyframes slideUp {
            from {
              opacity: 0;
              transform: translateY(30px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
          
          h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
          }
          
          .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1em;
          }
          
          .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            margin-bottom: 30px;
            transition: all 0.3s ease;
            cursor: pointer;
          }
          
          .upload-area:hover {
            border-color: #764ba2;
            background: linear-gradient(135deg, #e0e7ff 0%, #c3cfe2 100%);
            transform: scale(1.02);
          }
          
          .upload-area.dragover {
            border-color: #764ba2;
            background: linear-gradient(135deg, #d4d9ff 0%, #a8b3e8 100%);
          }
          
          #file-input {
            display: none;
          }
          
          .file-label {
            display: inline-block;
            padding: 15px 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 50px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
          }
          
          .file-label:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
          }
          
          .file-name {
            margin-top: 15px;
            color: #555;
            font-weight: 500;
          }
          
          .predict-btn {
            display: block;
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            border-radius: 50px;
            font-size: 1.2em;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
            margin-top: 20px;
          }
          
          .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(245, 87, 108, 0.6);
          }
          
          .predict-btn:active {
            transform: translateY(0);
          }
          
          .predict-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
          }
          
          #result {
            margin-top: 40px;
          }
          
          .loading {
            text-align: center;
            padding: 40px;
            color: #667eea;
            font-size: 1.2em;
          }
          
          .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
          }
          
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          
          .result-container {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 30px;
            animation: fadeIn 0.5s ease-out;
          }
          
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
          
          .preview-image {
            width: 100%;
            max-width: 400px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            margin: 20px auto;
            display: block;
          }
          
          .predictions-title {
            color: #667eea;
            margin: 30px 0 20px 0;
            font-size: 1.8em;
            text-align: center;
          }
          
          .prediction-list {
            list-style: none;
            padding: 0;
          }
          
          .prediction-item {
            background: white;
            margin: 15px 0;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s ease;
            animation: slideIn 0.4s ease-out;
            animation-fill-mode: both;
          }
          
          @keyframes slideIn {
            from {
              opacity: 0;
              transform: translateX(-20px);
            }
            to {
              opacity: 1;
              transform: translateX(0);
            }
          }
          
          .prediction-item:nth-child(1) { animation-delay: 0.1s; }
          .prediction-item:nth-child(2) { animation-delay: 0.2s; }
          .prediction-item:nth-child(3) { animation-delay: 0.3s; }
          .prediction-item:nth-child(4) { animation-delay: 0.4s; }
          .prediction-item:nth-child(5) { animation-delay: 0.5s; }
          
          .prediction-item:hover {
            transform: translateX(10px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
          }
          
          .prediction-item.top {
            border-left: 5px solid #f5576c;
            background: linear-gradient(90deg, #fff5f5 0%, white 100%);
          }
          
          .prediction-name {
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
          }
          
          .prediction-prob {
            font-size: 1.5em;
            font-weight: 700;
            color: #667eea;
          }
          
          .prediction-badge {
            display: inline-block;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-left: 10px;
            font-weight: 600;
          }
          
          .error {
            background: #fee;
            color: #c33;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #c33;
            margin-top: 20px;
          }
          
          .analysis-container {
            margin-top: 40px;
            animation: fadeIn 0.5s ease-out;
          }
          
          .analysis-title {
            color: #667eea;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2em;
            font-weight: 700;
          }
          
          .processing-step {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8eaf6 100%);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            animation: slideInFromLeft 0.6s ease-out;
            animation-fill-mode: both;
            border-left: 5px solid #667eea;
            transition: all 0.3s ease;
          }
          
          .processing-step:hover {
            transform: translateX(10px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
          }
          
          @keyframes slideInFromLeft {
            from {
              opacity: 0;
              transform: translateX(-50px);
            }
            to {
              opacity: 1;
              transform: translateX(0);
            }
          }
          
          .step-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            gap: 15px;
          }
          
          .step-number {
            font-size: 2em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 60px;
            height: 60px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
          }
          
          .step-title {
            color: #333;
            font-size: 1.5em;
            font-weight: 600;
            margin: 0;
          }
          
          .step-image {
            width: 100%;
            max-width: 500px;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            margin: 15px auto;
            display: block;
            animation: imageFadeIn 0.5s ease-out;
          }
          
          .histogram-img {
            max-width: 600px;
            background: white;
            padding: 10px;
          }
          
          @keyframes imageFadeIn {
            from {
              opacity: 0;
              transform: scale(0.95);
            }
            to {
              opacity: 1;
              transform: scale(1);
            }
          }
          
          .stats-box {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-top: 15px;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
          }
          
          .stat-item {
            padding: 10px;
            background: linear-gradient(135deg, #f5f7fa 0%, #e8eaf6 100%);
            border-radius: 8px;
            font-size: 1em;
            color: #555;
          }
          
          .stat-item strong {
            color: #667eea;
            margin-right: 8px;
          }
          
          .predictions-section {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 30px;
            margin-top: 30px;
            animation: fadeIn 0.8s ease-out;
          }
          
          .container {
            max-width: 1200px;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>🩺 Prescription Classifier</h1>
          <p class="subtitle">Upload a handwritten prescription image to identify the medicine</p>
          
          <div class="upload-area" id="upload-area">
            <label for="file-input" class="file-label">
              📁 Choose Image File
            </label>
          <input type="file" id="file-input" name="file" accept="image/*" />
            <div class="file-name" id="file-name">No file selected</div>
          </div>
          
          <button class="predict-btn" id="predict-btn" onclick="upload()" disabled>🔍 Predict Medicine</button>
          
        <div id="result"></div>
        </div>
        
        <script>
          const fileInput = document.getElementById('file-input');
          const fileName = document.getElementById('file-name');
          const predictBtn = document.getElementById('predict-btn');
          const uploadArea = document.getElementById('upload-area');
          const resultDiv = document.getElementById('result');
          
          fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
              fileName.textContent = '📄 ' + e.target.files[0].name;
              predictBtn.disabled = false;
            } else {
              fileName.textContent = 'No file selected';
              predictBtn.disabled = true;
            }
          });
          
          uploadArea.addEventListener('click', () => fileInput.click());
          
          uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
          });
          
          uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
          });
          
          uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
              fileInput.files = e.dataTransfer.files;
              fileName.textContent = '📄 ' + e.dataTransfer.files[0].name;
              predictBtn.disabled = false;
            }
          });
          
          async function upload(){
            const fi = document.getElementById('file-input');
            if(!fi.files.length){ 
              alert('Please choose an image first!');
              return; 
            }
            
            const file = fi.files[0];
            const fd = new FormData();
            fd.append('file', file);
            
            resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Starting image analysis...</p></div>';
            predictBtn.disabled = true;
            
            try {
              const resp = await fetch('/process_and_predict', { method: 'POST', body: fd });
              const data = await resp.json();
              
              if (data.error) {
                resultDiv.innerHTML = `<div class="error">❌ Error: ${data.error}</div>`;
                predictBtn.disabled = false;
                return;
              }
              
              const steps = data.processing_steps;
              const stepsList = [
                { key: 'original', title: '1. Original Image', icon: '🖼️' },
                { key: 'noise_reduction', title: '2. Noise Reduction', icon: '🔇' },
                { key: 'enhancement', title: '3. Image Enhancement', icon: '✨' },
                { key: 'histogram', title: '4. Histogram Analysis', icon: '📊' },
                { key: 'binary_threshold', title: '5. Binary Thresholding', icon: '⚫' },
                { key: 'morphological', title: '6. Morphological Analysis', icon: '🔬' },
                { key: 'segmentation', title: '7. Image Segmentation', icon: '✂️' }
              ];
              
              let out = '<div class="analysis-container">';
              out += '<h2 class="analysis-title">📋 Image Processing Pipeline</h2>';
              
              // Display each processing step
              stepsList.forEach((step, index) => {
                const stepData = steps[step.key];
                if (stepData) {
                  out += `<div class="processing-step" style="animation-delay: ${index * 0.2}s">`;
                  out += `<div class="step-header">`;
                  out += `<span class="step-number">${step.icon}</span>`;
                  out += `<h3 class="step-title">${step.title}</h3>`;
                  out += `</div>`;
                  
                  if (step.key === 'histogram') {
                    out += `<img src="${stepData}" class="step-image histogram-img" alt="${step.title}" />`;
                    if (data.histogram_stats) {
                      const stats = data.histogram_stats;
                      out += `<div class="stats-box">`;
                      out += `<div class="stat-item"><strong>Mean:</strong> ${stats.mean.toFixed(2)}</div>`;
                      out += `<div class="stat-item"><strong>Std Dev:</strong> ${stats.std.toFixed(2)}</div>`;
                      out += `<div class="stat-item"><strong>Min:</strong> ${stats.min}</div>`;
                      out += `<div class="stat-item"><strong>Max:</strong> ${stats.max}</div>`;
                      out += `</div>`;
                    }
                  } else {
                    out += `<img src="${stepData}" class="step-image" alt="${step.title}" />`;
                  }
                  out += `</div>`;
                }
              });
              
              // Final predictions
              out += '<div class="predictions-section">';
              out += '<h2 class="predictions-title">🎯 Final Predictions</h2>';
              out += '<ul class="prediction-list">';
              
              data.predictions.forEach((p, index) => {
                const isTop = index === 0;
                const topClass = isTop ? ' top' : '';
                out += `<li class="prediction-item${topClass}">`;
                out += `<span class="prediction-name">${p.class_name}${isTop ? ' <span class="prediction-badge">TOP MATCH</span>' : ''}</span>`;
                out += `<span class="prediction-prob">${(p.probability*100).toFixed(2)}%</span>`;
                out += '</li>';
              });
              
              out += '</ul></div></div>';
              resultDiv.innerHTML = out;
              predictBtn.disabled = false;
              
              // Scroll to results
              resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } catch (error) {
              resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
              predictBtn.disabled = false;
            }
          }
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/process_and_predict")
async def process_and_predict(file: UploadFile = File(...)):
    """Complete pipeline: process image through all steps then predict"""
    contents = await file.read()
    try:
        # Process image through all preprocessing steps
        processing_results = process_image_pipeline(contents)
        
        # Prepare final processed image for model
        final_img = processing_results['final_processed']
        
        # Convert to bytes for model input
        final_img_pil = Image.fromarray(final_img.astype(np.uint8))
        buf = io.BytesIO()
        final_img_pil.save(buf, format='PNG')
        processed_bytes = buf.getvalue()
        
        # Preprocess for model
        x = preprocess_image_bytes(processed_bytes)
        
        # Get predictions
        preds = model.predict(x, verbose=0)
        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)
        probs = preds[0]

        top_k = min(TOP_K, len(probs))
        top_idxs = probs.argsort()[-top_k:][::-1]

        results = []
        for i in top_idxs:
            cls_name = idx_to_class.get(str(i), str(i))
            results.append({
                "class_index": int(i),
                "class_name": cls_name,
                "probability": float(probs[i]),
                "is_top": bool(i == top_idxs[0])  # Convert numpy bool to Python bool
            })

        # Clean up processing_results to remove numpy arrays and ensure JSON serializability
        clean_processing_results = {}
        for key, value in processing_results.items():
            if key == 'final_processed':
                # Skip final_processed as it's a numpy array (already used for prediction)
                continue
            elif key == 'histogram_stats':
                # Skip histogram_stats - we'll handle it separately
                continue
            else:
                # Keep base64 strings as they are
                clean_processing_results[key] = value

        # Get histogram stats separately and convert to native Python types
        hist_stats = processing_results.get('histogram_stats', {})
        clean_hist_stats = {}
        if hist_stats and isinstance(hist_stats, dict):
            clean_hist_stats = {
                'mean': float(hist_stats.get('mean', 0)),
                'std': float(hist_stats.get('std', 0)),
                'min': int(hist_stats.get('min', 0)),
                'max': int(hist_stats.get('max', 0))
            }

        return JSONResponse({
            "predictions": results,
            "processing_steps": clean_processing_results,
            "histogram_stats": clean_hist_stats
        })
    except Exception as e:
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=400)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Simple prediction without processing steps"""
    contents = await file.read()
    try:
        x = preprocess_image_bytes(contents)
    except Exception as e:
        return JSONResponse({"error": f"Could not process image: {str(e)}"}, status_code=400)

    preds = model.predict(x, verbose=0)
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)
    probs = preds[0]

    top_k = min(TOP_K, len(probs))
    top_idxs = probs.argsort()[-top_k:][::-1]

    results = []
    for i in top_idxs:
        cls_name = idx_to_class.get(str(i), str(i))
        results.append({
            "class_index": int(i),
            "class_name": cls_name,
            "probability": float(probs[i]),
            "is_top": True
        })

    try:
        preview = Image.open(io.BytesIO(contents)).convert('RGB')
        preview.thumbnail((400, 400))
        buf = io.BytesIO()
        preview.save(buf, format='JPEG')
        data_url = "data:image/jpeg;base64," + (base64.b64encode(buf.getvalue()).decode('ascii'))
    except Exception:
        data_url = None

    return JSONResponse({
        "predictions": results,
        "preview": data_url,
    })


@app.post('/noise_reduction')
async def endpoint_noise_reduction(
    file: UploadFile = File(...),
    method: str = Form("gaussian"),
    kernel_size: int = Form(5)
):
    """Apply noise reduction to image
    
    Methods: 'gaussian', 'median', 'bilateral'
    """
    contents = await file.read()
    try:
        img = load_image_from_bytes(contents)
        result = noise_reduction(img, method=method, kernel_size=kernel_size)
        result_base64 = image_to_base64(result)
        return JSONResponse({
            "processed_image": result_base64,
            "method": method,
            "kernel_size": kernel_size
        })
    except Exception as e:
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=400)


@app.post('/enhancement')
async def endpoint_enhancement(
    file: UploadFile = File(...),
    brightness: float = Form(20.0),
    contrast: float = Form(1.2)
):
    """Enhance image brightness and contrast
    
    Brightness: -255 to 255 (0 = no change)
    Contrast: 0.0 to 3.0 (1.0 = no change)
    """
    contents = await file.read()
    try:
        img = load_image_from_bytes(contents)
        result = enhance_brightness_contrast(img, brightness=brightness, contrast=contrast)
        result_base64 = image_to_base64(result)
        return JSONResponse({
            "processed_image": result_base64,
            "brightness": brightness,
            "contrast": contrast
        })
    except Exception as e:
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=400)


@app.post('/binary_threshold')
async def endpoint_binary_threshold(
    file: UploadFile = File(...),
    method: str = Form("otsu"),
    threshold_value: int = Form(127)
):
    """Apply binary thresholding to image
    
    Methods: 'otsu', 'adaptive', 'simple'
    """
    contents = await file.read()
    try:
        img = load_image_from_bytes(contents)
        result = binary_thresholding(img, method=method, threshold_value=threshold_value)
        result_base64 = image_to_base64(result)
        return JSONResponse({
            "processed_image": result_base64,
            "method": method,
            "threshold_value": threshold_value if method == "simple" else None
        })
    except Exception as e:
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=400)


@app.post('/morphological')
async def endpoint_morphological(
    file: UploadFile = File(...),
    operation: str = Form("opening"),
    kernel_size: int = Form(5),
    iterations: int = Form(1)
):
    """Apply morphological operations to image
    
    Operations: 'opening', 'closing', 'erosion', 'dilation', 'gradient'
    """
    contents = await file.read()
    try:
        img = load_image_from_bytes(contents)
        result = morphological_operations(img, operation=operation, kernel_size=kernel_size, iterations=iterations)
        result_base64 = image_to_base64(result)
        return JSONResponse({
            "processed_image": result_base64,
            "operation": operation,
            "kernel_size": kernel_size,
            "iterations": iterations
        })
    except Exception as e:
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=400)


@app.post('/segmentation')
async def endpoint_segmentation(
    file: UploadFile = File(...),
    method: str = Form("watershed")
):
    """Apply image segmentation
    
    Methods: 'watershed', 'contour'
    """
    contents = await file.read()
    try:
        img = load_image_from_bytes(contents)
        result = image_segmentation(img, method=method)
        result_base64 = image_to_base64(result)
        return JSONResponse({
            "processed_image": result_base64,
            "method": method
        })
    except Exception as e:
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=400)


@app.post('/histogram_analysis')
async def endpoint_histogram_analysis(file: UploadFile = File(...)):
    """Perform histogram analysis on image"""
    contents = await file.read()
    try:
        img = load_image_from_bytes(contents)
        hist_data = histogram_analysis(img)
        hist_image_base64 = histogram_to_base64(hist_data['histogram'], bins=256, title="Image Histogram")
        
        return JSONResponse({
            "histogram_image": hist_image_base64,
            "statistics": {
                "mean": hist_data['mean'],
                "std": hist_data['std'],
                "min": hist_data['min'],
                "max": hist_data['max']
            }
        })
    except Exception as e:
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=400)


@app.post('/predict_batch')
async def predict_batch(files: list[UploadFile] = File(...)):
    outputs = []
    for file in files:
        contents = await file.read()
        try:
            x = preprocess_image_bytes(contents)
        except Exception as e:
            outputs.append({"filename": file.filename, "error": str(e)})
            continue
        preds = model.predict(x, verbose=0)
        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)
        probs = preds[0]
        top_idxs = probs.argsort()[-TOP_K:][::-1]
        results = [{"class_index": int(i), "class_name": idx_to_class.get(str(i), str(i)), "probability": float(probs[i])} for i in top_idxs]
        outputs.append({"filename": file.filename, "predictions": results})
    return JSONResponse(outputs)


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
