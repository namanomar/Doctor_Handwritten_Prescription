import numpy as np
import cv2
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def image_to_base64(img_array, format='PNG'):
    """Convert numpy array image to base64 string"""
    if isinstance(img_array, np.ndarray):
        img = Image.fromarray(img_array)
    else:
        img = img_array
    
    buf = io.BytesIO()
    img.save(buf, format=format)
    img_str = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/{format.lower()};base64,{img_str}"


def histogram_to_base64(hist_data, bins, title="Histogram"):
    """Create histogram image and return as base64"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(hist_data, bins=bins, color='#667eea', alpha=0.7, edgecolor='black')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    img_str = base64.b64encode(buf.getvalue()).decode('ascii')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


def load_image_from_bytes(image_bytes):
    """Load image from bytes and convert to numpy array"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return np.array(img)


def enhance_brightness_contrast(img, brightness=0, contrast=1.0):
    """Enhance image brightness and contrast"""
    img_float = img.astype(np.float32)
    
    # Brightness adjustment
    img_float = img_float + brightness
    
    # Contrast adjustment
    img_float = (img_float - 127.5) * contrast + 127.5
    
    # Clip values to valid range
    img_float = np.clip(img_float, 0, 255)
    
    return img_float.astype(np.uint8)


def noise_reduction(img, method='gaussian', kernel_size=5):
    """Apply noise reduction to image"""
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    elif method == 'median':
        return cv2.medianBlur(img, kernel_size)
    elif method == 'bilateral':
        return cv2.bilateralFilter(img, 9, 75, 75)
    else:
        return img


def binary_thresholding(img, method='otsu', threshold_value=127):
    """Apply binary thresholding"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    if method == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    elif method == 'simple':
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        binary = gray
    
    # Convert back to RGB for display
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def morphological_operations(img, operation='closing', kernel_size=5, iterations=1):
    """Apply morphological operations"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == 'opening':
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'closing':
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == 'erosion':
        result = cv2.erode(gray, kernel, iterations=iterations)
    elif operation == 'dilation':
        result = cv2.dilate(gray, kernel, iterations=iterations)
    elif operation == 'gradient':
        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    else:
        result = gray
    
    # Convert back to RGB for display
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


def image_segmentation(img, method='watershed'):
    """Apply image segmentation"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb = img.copy()
    else:
        gray = img
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    if method == 'watershed':
        try:
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Apply watershed - markers needs to be int32
            markers = markers.astype(np.int32)
            markers = cv2.watershed(img_rgb, markers)
            
            # Create visualization
            result = img_rgb.copy()
            result[markers == -1] = [255, 0, 0]  # Mark boundaries in red
            
            return result
        except Exception:
            # Fallback to contour method if watershed fails
            method = 'contour'
    
    if method == 'contour':
        # Threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        result = img_rgb.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        return result
    
    return img_rgb


def histogram_analysis(img):
    """Perform histogram analysis"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_flat = hist.flatten()
    
    # Calculate statistics and convert to native Python types
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    
    return {
        'histogram': hist_flat.tolist(),  # Convert numpy array to list
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val
    }


def process_image_pipeline(image_bytes):
    """Complete image processing pipeline"""
    # Load original image
    original_img = load_image_from_bytes(image_bytes)
    
    results = {}
    
    # Step 1: Original
    results['original'] = image_to_base64(original_img)
    
    # Step 2: Noise Reduction
    denoised = noise_reduction(original_img, method='gaussian')
    results['noise_reduction'] = image_to_base64(denoised)
    
    # Step 3: Brightness and Contrast Enhancement
    enhanced = enhance_brightness_contrast(denoised, brightness=20, contrast=1.2)
    results['enhancement'] = image_to_base64(enhanced)
    
    # Step 4: Histogram Analysis
    hist_data = histogram_analysis(enhanced)
    results['histogram'] = histogram_to_base64(hist_data['histogram'], bins=256, title="Image Histogram")
    # Only keep statistics, not the histogram array
    results['histogram_stats'] = {
        'mean': hist_data['mean'],
        'std': hist_data['std'],
        'min': hist_data['min'],
        'max': hist_data['max']
    }
    
    # Step 5: Binary Thresholding
    binary = binary_thresholding(enhanced, method='otsu')
    results['binary_threshold'] = image_to_base64(binary)
    
    # Step 6: Morphological Operations
    morph = morphological_operations(binary, operation='opening', kernel_size=5)
    results['morphological'] = image_to_base64(morph)
    
    # Step 7: Segmentation
    # segmented = image_segmentation(enhanced, method='watershed')
    # results['segmentation'] = image_to_base64(segmented)
    
    # Final processed image for model prediction
    results['final_processed'] = enhanced
    
    return results

