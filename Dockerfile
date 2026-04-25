FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ARG GEMINI_API_KEY=""
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
ENV GOOGLE_API_KEY=${GEMINI_API_KEY}

# Install deps first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Initialise the database at build time
RUN python -m scripts.init_db

# HF Spaces exposes port 7860
EXPOSE 7860

CMD ["bash", "-lc", "if [ -z \"$GOOGLE_API_KEY\" ] && [ -n \"$GEMINI_API_KEY\" ]; then export GOOGLE_API_KEY=\"$GEMINI_API_KEY\"; fi; uvicorn main:app --host 0.0.0.0 --port 7860"]
