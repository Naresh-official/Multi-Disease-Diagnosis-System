FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by TensorFlow, umap, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Make start script executable
RUN chmod +x start.sh

# Run the Streamlit app via start.sh (reads PORT env var)
CMD ["./start.sh"]
