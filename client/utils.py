import asyncio
import copy
import json
import torch
import io
import requests
import os
import websockets

from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm
from src.helper import get_device_id
from src.dataset import LoadDatasetEval, LoadStrategyB
from src.local_training import LocalTraining
from src.metrics import calc_accuracy, AverageMeter


device_id = -1
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    device_id = get_device_id(torch.cuda.is_available())
device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")


def get_config():
    with open("config.json", "r") as f:
        config = json.load(f)
    return config


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


def send_model(model: OrderedDict):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    address = os.getenv("PPHAR_SERVER_HOST")
    port = int(os.getenv("PPHAR_SERVER_PORT"))
    send_message(address=address, port=port,data=to_bytes(content=model))
    message = "Sending the local model to " + address
    print(message)
    loop.run_until_complete(send_log(message))
    return "Sent the local model"


def train(global_model):
    configuration = get_config()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loss_train = []
    w_local = None
    loss_locals = []
    local_valid_acc = []
    local = LocalTraining()
    w, loss, valid_acc = local.train(model=copy.deepcopy(global_model).to(device))
    w_local = w
    loss_locals.append(copy.deepcopy(loss))
    local_valid_acc.append(valid_acc)
    loss_avg = sum(loss_locals) / len(loss_locals)
    loss_train.append(loss_avg)
    accuracy_avg = sum(local_valid_acc) / len(local_valid_acc)
    message = 'Average loss = {:.3f} | Average accuracy = {:.3f}'.format(loss_avg, accuracy_avg)
    print(message)
    loop.run_until_complete(send_log(message))
    message = '@{},{:.3f},{:.3f}'.format(os.getenv("PPHAR_SUBJECT_ID"), loss_avg, accuracy_avg)
    loop.run_until_complete(send_log(message))
    torch.save(global_model, configuration["federated_parameters"]["fed_model_save"])
    return w_local

def test(global_model):
    global_model.eval()
    eval_acc_epoch = AverageMeter()
    configuration = get_config()
    data_processing_params = configuration["data_processing_params"]
    dataset_params = configuration["dataset_params"]

    load_obj = LoadDatasetEval(
        dataset_params["eval_src"],
        data_processing_params["seq_length"],
        dataset_params["eval_subject"],
        data_processing_params["overlap"],
        LoadStrategyB()
    )
    eval_data_loader = load_obj.prepare_eval_data_loader(dataset_params["eval_batch_size"])

    for (_, batch) in enumerate(eval_data_loader):
        X = batch['features']
        y = batch['labels']
        with torch.no_grad():
            pred = global_model(X)
        acc = calc_accuracy(pred.data,y.data)
        eval_acc_epoch.update(acc,X.size(0))


async def send_log(message: str):
    uri = "ws://172.17.0.1:7777"
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    message = dt + " - [" + os.getenv("PPHAR_CORE_ID") + "] " + message
    async with websockets.connect(uri) as websocket:
        await websocket.send(message)
