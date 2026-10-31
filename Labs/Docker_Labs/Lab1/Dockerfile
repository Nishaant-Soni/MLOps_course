# Use a slim official Python runtime, pinned to a patch release so rebuilds match
FROM python:3.11.18-slim

LABEL maintainer="Nishaant Soni" \
      description="Trains and compares classifiers on the scikit-learn Wine dataset"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install pinned dependencies first so this layer is cached when only code changes
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user that can write only to the artifacts directory.
# Done before copying the code so editing main.py doesn't rerun this layer.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts \
    && chown appuser:appuser /app/artifacts

# Copy the training script; it stays owned by root and read-only to appuser
COPY src/main.py .

USER appuser

# Model and metrics are written here; mount a host directory to keep them
VOLUME ["/app/artifacts"]

# Arguments passed to `docker run` replace the default CMD
ENTRYPOINT ["python", "main.py"]
CMD ["--output-dir", "/app/artifacts"]
