import copy
import json
import torch
import io
import requests

from typing import OrderedDict
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


def get_config(key: str):
    with open("config.json", "r") as f:
        config = json.load(f)
    return config[key]


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
    address = "server.pphar.io"
    port = 5000
    send_message(address=address, port=port,data=to_bytes(content=model))
    print("Sending the local model to " + address)
    return "Sent the local model"


def train(global_model):
    loss_train = []
    w_local = None
    for i in tqdm(range(get_config(key="epochs"))):
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
        print('Local Round = {:3d}/{} | Average loss = {:.3f} | Average accuracy = {:.3f}'.format(i, get_config(key="epochs"), loss_avg, accuracy_avg))
        torch.save(global_model, get_config(key="fed_model_save"))
    
    return w_local

def test(global_model):
    global_model.eval()
    eval_acc_epoch = AverageMeter()

    load_obj = LoadDatasetEval(
        get_config(key="eval_src"),
        get_config(key="seq_length"),
        get_config(key="eval_subject"),
        get_config(key="overlap"),
        LoadStrategyB()
    )
    eval_data_loader = load_obj.prepare_eval_data_loader(get_config(key="eval_batch_size"))

    for (_, batch) in enumerate(eval_data_loader):
        X = batch['features']
        y = batch['labels']
        with torch.no_grad():
            pred = global_model(X)
        acc = calc_accuracy(pred.data,y.data)
        eval_acc_epoch.update(acc,X.size(0))
