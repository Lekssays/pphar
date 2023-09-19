from flask import Flask, request
from tee_utils import *

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2048 * 1024 * 1024

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
    # w_global = initiliaze_global_model()
    # send_global_model(model=w_global, init=True, encrypted=False)
    return "initialized the tee server\n"


@app.route("/failed", methods = ['GET', 'POST'])
def process_failed_containers():
    if request.method == 'GET':
        return "<p>Hello, World!</p>"
    if request.method == 'POST':
        response = process_failed_request(request=request)
        return response


@app.route("/resume", methods = ['GET'])
def process_resume_containers():
    response = process_resume_request(request=request)
    return response


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5555, threads=0)
