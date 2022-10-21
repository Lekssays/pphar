import threading

from flask import Flask, request

from utils import *

from src.main_fed import FedAvg


w_locals = []
sem = threading.Semaphore()

app = Flask(__name__)

@app.route("/models", methods = ['GET', 'POST'])
def update():
    if request.method == 'GET':
        return "<p>Hello, World!</p>"
    if request.method == 'POST':
        data = request.get_data()
        response = process_request(data=data)
        return response


@app.route("/init", methods = ['GET'])
def init():
    w_global = initiliaze_global_model()
    send_global_model(w_global)
    return "<p>initialized the model</p>"


def process_request(data):
    model = from_bytes(data)
    if len(w_locals) < get_config(key="n_subjects") - 1:
        print(len(w_locals), type(get_config(key="n_subjects")), get_config(key="n_subjects"))
        w_locals.append(model)
        print("Received a local model.")
        return "Received a local model."
    else:
        print("Aggregating local models.")
        if not get_config(key="encrypted"):
            w_global = FedAvg(w_locals)
        w_locals.clear()
        send_global_model(model=w_global)
        print("Sent aggregated global model.")
        return "Sent aggregated global model."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
