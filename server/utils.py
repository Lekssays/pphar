import torch
import io
import requests
import json
import threading

from collections import OrderedDict

from src.main_fed import FedAvg
from src.SingleLSTM import SingleLSTMEncoder
from helper import get_device_id

device_id = -1
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    device_id = get_device_id(torch.cuda.is_available())
device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")


def get_config(key: str):
    with open("config.json", "r") as f:
        config = json.load(f)
    return config[key]


def initiliaze_global_model():
    n_channels = get_config(key="n_channels")
    n_hidden_layers = get_config(key="n_hidden_layers")
    n_layers = get_config(key="n_layers")
    n_classes = get_config(key="n_classes")
    drop_prob = get_config(key="drop_prob")
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


def send_global_model(model: OrderedDict, rounds: int):
    subjects = get_config(key="subjects")
    for s in subjects:
        address = "subject" + str(s) + ".pphar.io"
        port = 5000
        data = to_bytes(content=model)
        print("Sending the global model to " + address + " / " + str(rounds))
        t = threading.Thread(target=send_message, args=(address, port, data,))
        t.start()
