import torch
import logging

from flask import Flask, request

from pkg.config import Config
from pkg.l2s import Landmarks2ScreenCoords

logging.getLogger('werkzeug').disabled = True

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('app')

device = 'cpu' if torch.cuda.device_count() == 0 else 'cuda'


l2coord = Landmarks2ScreenCoords(logger)

logger.info(f'Using device {device}')
app = Flask('l2s', static_url_path='/', static_folder='static')
@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')


@app.route('/api', methods=['GET', 'POST', 'HEAD'])
def api_index():
    return '<HTML><HEAD></HEAD><BODY>This is the app API</BODY></HTML>'


@app.route('/api/gaze/train/<int:epochs>', methods=['POST'])
def train(epochs):
    return l2coord.train(epochs)


@app.route('/api/gaze/pca', methods=['POST'])
def pca():

    return l2coord.do_pca()


@app.route('/api/gaze/config', methods=['POST'])
def config():
    return {'config': Config().__dict__}


@app.route('/api/gaze/landmark-features', methods=['POST'])
def landmark_features():

    landmarks_and_target = request.json

    feats = landmarks_and_target["landmarks"]
    target = landmarks_and_target["target"]

    if target["target_x"] != "undefined":
        target = [float(target["target_x"]), float(target["target_y"])]
    else:
        target = None

    # Send to landmark model
    # print("---------------------------------------------------")
    # print(feats)
    # print("---------------------------------------------------")
    return l2coord.predict(feats, target)


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
        l2coord.save()
        return {'status': 'success'}
    except Exception as e:
        app.logger.info(f'POST /api/gaze/save: {e}')
        return {'status': 'failed'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
