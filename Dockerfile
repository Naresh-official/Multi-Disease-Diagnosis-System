FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Make start script executable
RUN chmod +x start.sh

# Run the Streamlit app via start.sh (reads PORT env var)
CMD ["./start.sh"]
