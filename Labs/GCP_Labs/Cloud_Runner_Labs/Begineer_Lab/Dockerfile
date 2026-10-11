FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Run as an unprivileged user instead of root.
RUN useradd --create-home appuser
USER appuser

# Cloud Run injects PORT at runtime; 8080 is the default for local runs.
ENV PORT=8080
EXPOSE 8080

# Production server. gunicorn binds to 0.0.0.0:$PORT by default, and threads
# let one instance handle concurrent requests. JSON form lets it receive
# SIGTERM directly for graceful shutdown.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
