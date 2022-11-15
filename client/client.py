import asyncio

from collections import OrderedDict
from flask import Flask, request
import opacus
from src.SingleLSTM import SingleLSTMEncoder
from src.DPLSTM import DPLSTMEncoder
from utils import *

app = Flask(__name__)


@app.route("/models", methods = ['GET', 'POST'])
def update():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if request.method == 'GET':
        return "<p>Hello, World!</p>"
    if request.method == 'POST':
        data = request.get_data()
        message = "Received a global model."
        print(message)
        loop.run_until_complete(send_log(message))
        _ = process_request(data=data)
        return "Received a global model."


def process_request(data):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
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
    loop.run_until_complete(send_log(message))
    w_local = train(global_model)
    send_model(model=w_local)
    torch.cuda.empty_cache()
    return w_local


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
