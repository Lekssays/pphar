import asyncio
import torch

from flask import Flask, request
<<<<<<< HEAD

=======
import opacus
from src.SingleLSTM import SingleLSTMEncoder
from src.DPLSTM import DPLSTMEncoder
>>>>>>> Fixed different learning rates for DP and normal.
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
<<<<<<< HEAD
    message = "Received an initial global model."
    print(message, flush=True)
=======
    w_global = from_bytes(data)
    if type(w_global) != OrderedDict:
        message = "The received global model is not an OrderedDict."
        print(message)
        loop.run_until_complete(send_log(message))
        return None
    drop_keys = ["lstm.l0.ih.weight", "lstm.l0.ih.bias", "lstm.l0.hh.weight", "lstm.l0.hh.bias"]
    n_channels = get_config(key="n_channels")
    n_hidden_layers = get_config(key="n_hidden_layers")
    n_layers = get_config(key="n_layers")
    n_classes = get_config(key="n_classes")
    drop_prob = get_config(key="drop_prob")
    subject = os.getenv("PPHAR_SUBJECT_ID")
    if int(subject) in get_config(key="dp_sgd_clients"):
        global_model = DPLSTMEncoder(n_channels, n_hidden_layers, n_layers, n_classes, drop_prob)
        for keys in w_global.keys():
            if keys not in drop_keys:
                global_model.state_dict()[keys] = w_global[keys]
    else:
        global_model = SingleLSTMEncoder(n_channels, n_hidden_layers, n_layers, n_classes, drop_prob)
        global_model.load_state_dict(w_global)
        
    message = "Training with the new global model"
    print(message)
>>>>>>> Fixed different learning rates for DP and normal.
    loop.run_until_complete(send_log(message))
    if get_config("encrypted"):
        print("encrypted init", flush=True)
        _ = process_encrypted_request(request=request, init=True)
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
    process_encryption_notification(request=request)
    return "Received"


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000, threads=10)
