# Use a slim official Python runtime as the parent image
FROM python:3.11-slim

LABEL maintainer="Nishaant Soni" \
      description="Trains and compares classifiers on the scikit-learn Wine dataset"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install pinned dependencies first so this layer is cached when only code changes
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script into the container
COPY src/main.py .

# Run as a non-root user that owns the artifacts directory
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts \
    && chown -R appuser:appuser /app
USER appuser

# Model and metrics are written here; mount a host directory to keep them
VOLUME ["/app/artifacts"]

# Arguments passed to `docker run` replace the default CMD
ENTRYPOINT ["python", "main.py"]
CMD ["--output-dir", "/app/artifacts"]
