import asyncio

from flask import Flask, request

from utils import *

app = Flask(__name__)


@app.route("/models", methods = ['GET', 'POST'])
def process_models():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if request.method == 'GET':
        return "<p>Hello, World!</p>"
    if request.method == 'POST':
        message = "Received a global model."
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        _ = process_request(request=request)
        return "Received a global model."


@app.route("/init", methods = ['POST'])
def init():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    message = "Received an initial global model."
    print(message, flush=True)
    loop.run_until_complete(send_log(message))
    if get_config("encrypted"):
        print("encrypted init", flush=True)
        _ = process_encrypted_request(request=request, init=True)
    else:
        _ = process_request(request=request)
    return message


@app.route("/enc_models", methods = ['GET', 'POST'])
def process_encrypted_models():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if request.method == 'GET':
        return "<p>Hello, World!</p>"
    if request.method == 'POST':
        message = "Received an encrypted global model."
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        _ = process_encrypted_request(request=request)
        return message

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
