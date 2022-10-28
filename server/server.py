from flask import Flask, request
from utils import *

app = Flask(__name__)

@app.route("/models", methods = ['GET', 'POST'])
def process_models():
    if request.method == 'GET':
        return "<p>Hello, World!</p>"
    if request.method == 'POST':
        response = process_request(request=request)
        return response


@app.route("/enc_models", methods = ['GET', 'POST'])
def process_encrypted_models():
    if request.method == 'GET':
        return "<p>Hello, World!</p>"
    if request.method == 'POST':
        response = process_encrypted_request(request=request)
        return response


@app.route("/init", methods = ['GET'])
def init():
    w_global = initiliaze_global_model()
    send_global_model(model=w_global, init=True, encrypted=False)
    return "initialized the model\n"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
