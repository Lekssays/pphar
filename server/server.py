import torch

from flask import Flask, request

from src.main_fed import FedAvg
from utils import *

w_locals = []

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
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    model = from_bytes(data)
    
    rounds = get_rounds()
    if rounds > get_config(key="epochs"):
        message = "Training finished :)"
        print(message)
        loop.run_until_complete(send_log(message))
        return "Training finished."

    if len(w_locals) < len(get_config(key="subjects")) - 1:
        w_locals.append(model)
        message = "Received a local model."
        print(message)
        loop.run_until_complete(send_log(message))   
        return "Received a local model."
    else:
        message = "Aggregating local models."
        print(message)
        loop.run_until_complete(send_log(message))   
        if not get_config(key="encrypted"):
            w_global = FedAvg(w_locals)
        w_locals.clear()
        send_global_model(model=w_global)
        message = "Sent aggregated global model."
        print(message)
        loop.run_until_complete(send_log(message))

        message = "Saving the aggregated global model locally."
        print(message)
        loop.run_until_complete(send_log(message))
        torch.save(w_global, "w_global.pt")

        return"Sent and saved aggregated global model."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
