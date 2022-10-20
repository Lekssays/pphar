from flask import Flask, request
from utils import *

from src.main_fed import FedAvg


w_locals = []
rounds = 0
app = Flask(__name__)

@app.route("/models", methods = ['GET', 'POST'])
def update():
    if request.method == 'GET':
        return "<p>Hello, World!</p>"
    if request.method == 'POST':
        data = request.get_data()
        response = process_request(data=data)
        send_global_model(response, 1)
        return response


@app.route("/init", methods = ['GET'])
def init():
    w_global = initiliaze_global_model()
    send_global_model(w_global, 1)


def process_request(data):
    model = from_bytes(data)
    if len(w_locals) < get_config(key="n_subjects"):
        w_locals.append(model)
        print("Received a local model.")
        return "Received a model"
    else:
        print("Aggregating local models.")
        if not get_config(key="encrypted"):
            w_global = FedAvg(w_locals)
        w_locals.clear()
        return to_bytes(w_global)


if __name__ == "__main__":
    init()
    app.run(host="0.0.0.0", port=5000, debug=True)
