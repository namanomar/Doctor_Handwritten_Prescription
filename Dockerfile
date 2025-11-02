# Use Python 3.10 slim image
FROM python:3.10-slim

# Install system dependencies for OpenCV and image processing
# Must be done as root before switching to user
# Note: Removed libgl1-mesa-glx (not available in Debian Trixie)
# OpenCV works headless without OpenGL for most operations
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Create app directories and set ownership
RUN mkdir -p /app/model /app/api && \
    chown -R user:user /app

# Switch to non-root user
USER user

# Set environment variables
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Copy requirements file first (for better Docker layer caching)
COPY --chown=user:user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=user:user api/ ./api/
COPY --chown=user:user model/ ./model/

# Expose port (Hugging Face Spaces uses port 7860)
EXPOSE 7860

# Run the application using uvicorn from local bin
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
