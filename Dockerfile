# Use python 3.11 slim as the base image
FROM python:3.11-slim

# Install system dependencies for OpenCV and CairoSVG
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose port 8501
EXPOSE 8501

# Set the entry point to run Streamlit
ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
