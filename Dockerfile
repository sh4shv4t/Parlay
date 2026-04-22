FROM python:3.11-slim

WORKDIR /app

# Install deps first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Initialise the database at build time
RUN python -m scripts.init_db

# startup script
RUN chmod +x scripts/start.sh

# HF Spaces exposes port 7860
EXPOSE 7860

CMD ["bash", "scripts/start.sh"]
