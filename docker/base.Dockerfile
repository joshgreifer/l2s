FROM python:3.7
# Run this from root of project
# e.g. docker build -t l2s:base -f docker\base.Dockerfile .
RUN apt-get update
COPY docker/scripts/install_python_deps.sh ./

RUN /bin/bash install_python_deps.sh
