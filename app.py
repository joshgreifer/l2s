import os

from flask import Flask, send_from_directory


app = Flask(__name__, static_url_path='/', static_folder='static')


@app.route('/', methods=['GET'])
def index():
    """Serve the main application page."""
    return app.send_static_file('index.html')


@app.route('/models/<path:filename>', methods=['GET'])
def download_model(filename):
    """Allow clients to download pre-trained model files."""
    model_dir = os.path.join(app.root_path, 'cache', 'checkpoints')
    return send_from_directory(model_dir, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

