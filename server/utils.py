import asyncio
import copy
import io
import json
import torch
import requests
import os
import websockets


from collections import OrderedDict
from datetime import datetime
from src.main_fed import FedAvg
from Pyfhel import Pyfhel, PyCtxt
from multiprocessing import Process
from src.SingleLSTM import SingleLSTMEncoder
from helper import get_device_id


device_id = -1
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    device_id = get_device_id(torch.cuda.is_available())
device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")

rounds = 0
w_locals = []


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
    torch.save(global_model, "w_global.pt")
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


def send_message(address: str, port: int, data: bytes, init: bool, encrypted: bool):
    if init:
        url = "http://" + address + ":" + str(port) + "/init"
    else:
        if encrypted:
            url = "http://" + address + ":" + str(port) + "/enc_models"
        else:
            url = "http://" + address + ":" + str(port) + "/models"
    payload = {
        "data": data.decode("cp437"),
    }
    return requests.post(url=url, json=payload, headers={'Content-Type': 'application/json'})


def send_global_model(model, init, encrypted):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    global rounds
    subjects = get_config(key="subjects")
    data = to_bytes(content=model)
    for s in subjects:
        address = "subject" + str(s) + ".pphar.io"
        port = 5000
        if encrypted:
            message = "Sending the encrypted global model to " + address + " / " + str(rounds)
        else:   
            if init:
                message = "Sending the initial global model to " + address + " / " + str(rounds)
            else:
                message = "Sending the global model to " + address + " / " + str(rounds)
        print(message)
        loop.run_until_complete(send_log(message))   
        p = Process(target=send_message, args=(address, port, data, init, encrypted, ))
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


def EncFedAvg(HE, enc_w):
    n = 1.0 / len(enc_w)
    enc_w_avg = copy.deepcopy(enc_w[0])
    for k in enc_w_avg.keys():
        for i in range(1, len(enc_w)):
            enc_w_avg[k] = enc_w[i][k] + enc_w_avg[k]
        enc_w_avg[k] = enc_w_avg[k] * n
        HE.relinKeyGen()
        HE.rescale_to_next(enc_w_avg[k])
        enc_w_avg[k] = enc_w_avg[k].to_bytes().decode('cp437')
    return enc_w_avg


def process_request(request):
    global w_locals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    model = from_bytes(request.json.get("data"))
    sender = request.json.get('sender')
    
    rounds = get_rounds()
    if rounds > get_config(key="epochs"):
        message = "Training finished :)"
        print(message)
        loop.run_until_complete(send_log(message))
        return "Training finished."

    if len(w_locals) < len(get_config(key="subjects")) - 1:
        w_locals.append(model)
        message = f"Received a local model from {sender}"
        print(message)
        loop.run_until_complete(send_log(message))   
        return message
    else:
        message = "Aggregating local models."
        print(message)
        loop.run_until_complete(send_log(message))   
        w_global = FedAvg(w_locals)
        w_locals.clear()

        message = "Saving the aggregated global model locally."
        print(message)
        loop.run_until_complete(send_log(message))
        torch.save(w_global, "w_global.pt")

        send_global_model(model=w_global, init=False, encrypted=False)
        print(w_global)
        message = f"Sent aggregated global model to {sender}"
        print(message)
        loop.run_until_complete(send_log(message))
    
        return message


def process_encrypted_request(request):
    global w_locals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    rounds = get_rounds()
    if rounds > get_config(key="epochs"):
        message = "Training finished :)"
        print(message)
        loop.run_until_complete(send_log(message))
        return "Training finished."

    HE = Pyfhel()
    HE.from_bytes_context(request.json.get('context').encode('cp437'))
    HE.from_bytes_public_key(request.json.get('pk').encode('cp437'))
    HE.from_bytes_relin_key(request.json.get('rlk').encode('cp437'))
    HE.from_bytes_rotate_key(request.json.get('rtk').encode('cp437'))
    cx = PyCtxt(pyfhel=HE, bytestring=request.json.get('data').encode('cp437'))
    sender = request.json.get('sender')
    print(f"Received HE={HE} and cx={cx} from {sender}")

    if len(w_locals) < len(get_config(key="subjects")) - 1:
        w_locals.append(cx)
        message = f"Received an encrypted local model from {sender}"
        print(message)
        loop.run_until_complete(send_log(message))
        return message
    else:
        message = f"Aggregating encrypted local models from {sender}"
        print(message)
        loop.run_until_complete(send_log(message))   
        enc_w_global = EncFedAvg(HE=HE, enc_w=w_locals)
        print(enc_w_global)
        w_locals.clear()
        send_global_model(model=enc_w_global, init=False, encrypted=True)
        message = f"Sent aggregated global model to {sender}"
        print(message)
        loop.run_until_complete(send_log(message))
        return enc_w_global
