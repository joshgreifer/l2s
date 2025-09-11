import json
import sys
import os
import torch
import logging

from flask import Flask, request

from pkg.config import Config
from pkg.l2s import L2S
from pkg.util import AttrDict





logging.getLogger('werkzeug').disabled = True

# logging.basicConfig(level=logging.DEBUG)

default_config = "cache/config.json"
try:
    c = json.load(open("cache/app.config.json"))
    config_file = c['config_file'] if 'config_file' in c else default_config
except FileNotFoundError:
    print("No cache/app.config.json found, using default config")
    with open("cache/app.config.json", "w") as f:
        json.dump({
            "config_file": default_config
        }, f)
    config_file = default_config


device = 'cpu' if torch.cuda.device_count() == 0 else 'cuda'


l2coord = L2S(Config(config_file))

print(f'Using device {device}')

app = Flask(__name__, static_url_path='/', static_folder='static')
@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')


@app.route('/api', methods=['GET', 'POST', 'HEAD'])
def api_index():
    return '<HTML><HEAD></HEAD><BODY>This is the app API</BODY></HTML>'


@app.route('/api/gaze/train/<int:epochs>', methods=['POST'])
def train(epochs):
    if l2coord.model is not None:
        return l2coord.train(epochs, streaming_mode=False)
    else:
        return l2coord.losses

@app.route('/api/gaze/calibrate/<int:epochs>', methods=['POST'])
def calibrate(epochs):
    if l2coord.model is not None:
        return l2coord.train(epochs, streaming_mode=True)
    else:
        return l2coord.losses


@app.route('/api/gaze/pca', methods=['POST'])
def pca():
    return l2coord.do_pca()


@app.route('/api/gaze/config', methods=['POST'])
def config():
    return {'config': Config().__dict__}


@app.route('/api/gaze/landmarks', methods=['POST'])
def landmarks_():

    landmarks_and_target = request.json

    landmarks = landmarks_and_target["landmarks"]
    target = landmarks_and_target["target"]

    if target["target_x"] != "undefined":
        target = [float(target["target_x"]), float(target["target_y"])]
    else:
        target = None

    # Send to landmark model
    # print("---------------------------------------------------")
    # print(landmarks)
    # print("---------------------------------------------------")
    return l2coord.predict(landmarks, target)


@app.route('/api/gaze/save', methods=['POST', 'HEAD', 'GET'])
def save_model():
    try:
        l2coord.save(1)
        return {'status': 'success'}
    except Exception as e:
        print(f'POST /api/gaze/save: {e}', file=sys.stderr)
        return {'status': 'failed'}


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, ssl_context=('cache/cert.pem', 'cache/key.pem'), debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
