import asyncio
import torch

from flask import Flask, request
from utils import *

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 2048 * 1024 * 1024

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
        process_request(request=request)
        torch.cuda.empty_cache()
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
        process_encrypted_request(request=request, init=True)
        torch.cuda.empty_cache()
    else:
        process_request(request=request)
        torch.cuda.empty_cache()
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
        process_encrypted_request(request=request)
        torch.cuda.empty_cache()
        return message


@app.route("/encrypting", methods = ['POST'])
def encrypting():
    process_memory_queue_action(request=request)
    return "Received"


if __name__ == "__main__":
    from waitress import serve
    #Here we have separate the clients sending to the server and the clients sending to TEE
    #These are clients sending to the TEE server
    client_number = int(os.environ["PPHAR_SUBJECT_ID"])
    print("Client Number",client_number,flush=True)
    if client_number in get_config("tee_clients"):
        print("Serve to the TEE server",flush=True)
        serve(app, host="0.0.0.0", port=5555, threads=10)
    else:
    #These are clients sending to the server
        print("Serve to the federated server",flush=True)
        serve(app, host="0.0.0.0", port=5000, threads=10)
