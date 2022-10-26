import asyncio
import io
import json
import torch
import requests
import os
import websockets


from collections import OrderedDict
from datetime import datetime

from multiprocessing import Process
from src.dp_sgd_network import SingleLSTMEncoder
from helper import get_device_id

device_id = -1
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    device_id = get_device_id(torch.cuda.is_available())
device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")

rounds = 0

def get_config():
    with open("config.json", "r") as f:
        config = json.load(f)
    return config


def initiliaze_global_model():

    configuration = get_config()

    network_params = configuration["models"]["lstm_model_dpsgd"]["network_params"]

    n_channels = network_params["n_channels"]
    n_hidden_layers = network_params["num_hidden"]
    n_layers = network_params["n_layers"]
    n_classes = network_params["n_classes"]
    drop_prob = network_params["keep_prob"]
    global_model = SingleLSTMEncoder(n_channels, n_hidden_layers, n_layers, n_classes, drop_prob)
    global_model.to(device)
    global_model.train()
    return global_model.state_dict()


def to_bytes(content: OrderedDict) -> bytes:
    buff = io.BytesIO()
    torch.save(content, buff)
    buff.seek(0)
    return buff.read()


def from_bytes(content: bytes) -> torch.Tensor:
    buff = io.BytesIO(content)
    loaded_content = torch.load(buff)
    return loaded_content


def send_message(address: str, port: int, data: bytes):
    url = "http://" + address + ":" + str(port) + "/models"
    res = requests.post(url=url, data=data, headers={'Content-Type': 'application/octet-stream'})
    return res.content


def send_global_model(model: OrderedDict):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    global rounds
    configuration = get_config()
    subjects = configuration["federated_parameters"]["subjects"]
    for s in subjects:
        address = "subject" + str(s) + ".pphar.io"
        port = 5000
        data = to_bytes(content=model)
        message = "Sending the global model to " + address + " / " + str(rounds)
        print(message)
        loop.run_until_complete(send_log(message))   
        p = Process(target=send_message, args=(address, port, data,))
        p.start()
    rounds += 1
    return rounds


async def send_log(message: str):
    uri = "ws://172.17.0.1:7777"
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    message = dt + " - [" + os.getenv("PPHAR_CORE_ID") + "] " + message
    async with websockets.connect(uri) as websocket:
        await websocket.send(message)


def get_rounds():
    global rounds
    return rounds
