# Use Python 3.10 as base image
FROM python:3.10-alpine

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    python3-dev

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs

# Initialize database
RUN python -c "from database import init_db; init_db()"

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "run.py"] 