# ClusterMind research artifact: CPU-only image (works on Intel and Apple Silicon Macs)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CM_MODELS_DIR=/app/models

# compilers for llama-cpp-python (built from source for the host CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git gawk gzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
# The project folder is bind-mounted at /app by docker-compose, so code,
# data, runs/ and results/ live on your Mac and persist between runs.
EXPOSE 8000
ENTRYPOINT ["/bin/bash", "/app/cm"]
CMD ["help"]
