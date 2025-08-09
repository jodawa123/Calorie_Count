# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including C++ compiler
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version for better wheel support
RUN pip install --upgrade pip

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with prefer-binary flag
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs

# Initialize database
RUN python -c "from database import init_db; init_db()"

# Expose port
EXPOSE 5000

# Run the application with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"] 