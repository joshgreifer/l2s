
import logging

from flask import Flask, request

from pkg import *
from pkg.l2s.l2s import Landmarks2ScreenCoords
from pkg.util.util import AttrDict

logging.getLogger('werkzeug').disabled = True

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('app')
app = Flask('l2s', static_url_path='/', static_folder='static')

config = Config()

l2coord = Landmarks2ScreenCoords(app.logger)

logger.info(f'Using device {l2coord.config.device}')


# KVStore(f'user.json').put("This is a long piece of text, with\nline breaks.", "intro")


# @app.errorhandler(Exception)
# def app_error(ex):
#     return {"exception": f"{ex}", "traceback": traceback.extract_stack().format()}, 500


@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')


@app.route('/api', methods=['GET', 'POST', 'HEAD'])
def api_index():
    return '<HTML><HEAD></HEAD><BODY>This is the app API</BODY></HTML>'


@app.route('/api/gaze/train/<int:epochs>', methods=['POST'])
def train(epochs):
    return {"loss": l2coord.train_with_dataset(epochs)}


@app.route('/api/gaze/landmarks', methods=['POST'])
def landmarks():

    landmarks_and_target = request.json

    feats = landmarks_and_target["landmarks"]
    target = landmarks_and_target["target"]

    if target["target_x"] != "undefined":
        target = [float(target["target_x"]), float(target["target_y"])]
    else:
        target = None

    # Send to landmark model
    return l2coord.process_landmarks(feats, target)

@app.route('/api/gaze/save', methods=['POST', 'HEAD', 'GET'])
def save_g2s_model():
    try:
        l2coord.save()
        return {'status': 'success'}
    except Exception as e:
        app.logger.info(f'POST /api/gaze/save: {e}')
        return {'status': 'failed'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
