from collections import OrderedDict
from multiprocessing import Process
from flask import Flask, request

from src.SingleLSTM import SingleLSTMEncoder
from utils import from_bytes, get_config, train, send_model

app = Flask(__name__)


@app.route("/models", methods = ['GET', 'POST'])
def update():
    if request.method == 'GET':
        return "<p>Hello, World!</p>"
    if request.method == 'POST':
        data = request.get_data()
        print("Received a global model.")
        p = Process(target=process_request, args=(data,))
        p.start()
        return "Received a global model."


def process_request(data):
    w_global = from_bytes(data)
    if type(w_global) != OrderedDict:
        print("The received global model is not an OrderedDict.")
        return None
    n_channels = get_config(key="n_channels")
    n_hidden_layers = get_config(key="n_hidden_layers")
    n_layers = get_config(key="n_layers")
    n_classes = get_config(key="n_classes")
    drop_prob = get_config(key="drop_prob")
    global_model = SingleLSTMEncoder(n_channels, n_hidden_layers, n_layers, n_classes, drop_prob)
    global_model.load_state_dict(w_global)
    print("Training with the new global model")
    w_local = train(global_model)
    send_model(model=w_local)
    return w_local


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
