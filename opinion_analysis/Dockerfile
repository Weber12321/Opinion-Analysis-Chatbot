# opinion_analysis Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install build dependencies first
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY  requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download zh_core_web_sm

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
