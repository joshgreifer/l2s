# L2S

This repository provides a lightweight Flask server that delivers the web
client for the Learn2See project.

## Run Locally

1. Create a Python environment (see `install_python_env.sh`).
2. Ensure you have the latest version of npm installed.
3. `npm install`
4. Run the server (see `run.sh`).
5. Point a browser at <http://localhost:5000>.

## Architecture

Model training and calibration now occur entirely in the client. The Flask
application serves only static files and, optionally, pre-trained models. Any
model files placed in `cache/checkpoints` can be downloaded via
`/models/<filename>`.

