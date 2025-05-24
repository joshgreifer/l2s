FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# 1) Remove any packaged libnvidia-ml stub
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY docker/scripts/install_python_deps.sh ./

RUN /bin/bash install_python_deps.sh
