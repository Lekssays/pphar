import asyncio
import copy
import json
import torch
import time
import random
import numpy as np
import io
import requests
import os
import websockets
import psutil
import gc
import opacus

from collections import OrderedDict
from datetime import datetime
from os import path
from src.helper import get_device_id
from src.dataset import LoadDatasetEval, LoadStrategyB
from src.local_training import LocalTraining
from src.metrics import calc_accuracy, AverageMeter
from src.SingleLSTM import SingleLSTMEncoder
from src.DPLSTM import DPLSTMEncoder
from Pyfhel import Pyfhel, PyCtxt


global device

memory_queue = []


def get_config(key: str):
    with open("config.json", "r") as f:
        config = json.load(f)
    return config[key]


def to_bytes(content: OrderedDict) -> bytes:
    buff = io.BytesIO()
    torch.save(content, buff)
    buff.seek(0)
    return buff.read()


def from_bytes(content: bytes) -> OrderedDict:
    buff = io.BytesIO(content)
    loaded_content = torch.load(buff)
    return loaded_content


def send_message(address: str, port: int, model: OrderedDict, HE=None):
    if bool(get_config(key="encrypted")):
        url = "http://" + address + ":" + str(port) + "/enc_models"
        payload = {
            'context': HE.to_bytes_context().decode('cp437'),
            'pk': HE.to_bytes_public_key().decode('cp437'),
            'rlk': HE.to_bytes_relin_key().decode('cp437'),
            'rtk': HE.to_bytes_rotate_key().decode('cp437'),
            'data': to_bytes(content=model).decode('cp437'),
            'sender': os.getenv("PPHAR_CORE_ID"),
        }
    else:
        url = "http://" + address + ":" + str(port) + "/models"
        payload = {
            'sender': os.getenv("PPHAR_CORE_ID"),
            'data': to_bytes(content=model).decode('cp437'),
        }
    res = requests.post(url, data=json.dumps(payload), timeout=None)
    del payload
    gc.collect()
    return res


def send_model(model: OrderedDict):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    address = os.getenv("PPHAR_SERVER_HOST")
    port = int(os.getenv("PPHAR_SERVER_PORT"))
    send_message(address=address, port=port,model=model)
    del model
    gc.collect()
    message = "Sending the local model to " + address
    print(message, flush=True)
    loop.run_until_complete(send_log(message))
    return "Sent the local model"


def train(global_model, rounds):
    device_id = get_device_id(torch.cuda.is_available())
    global device
    device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loss_train = []
    loss_locals = []
    local_valid_acc = []
    local = LocalTraining(device)
    w_local, loss, valid_acc = local.train(model=copy.deepcopy(global_model).to(device))
    loss_locals.append(copy.deepcopy(loss))
    local_valid_acc.append(valid_acc)
    loss_avg = sum(loss_locals) / len(loss_locals)
    loss_train.append(loss_avg)
    accuracy_avg = sum(local_valid_acc) / len(local_valid_acc)
    message = 'Average loss = {:.3f} | Average accuracy = {:.3f}'.format(loss_avg, accuracy_avg)
    print(message, flush=True)
    loop.run_until_complete(send_log(message))
    message = '@{},{},{:.3f},{:.3f}'.format(rounds, os.getenv("PPHAR_SUBJECT_ID"), loss_avg, accuracy_avg)
    loop.run_until_complete(send_log(message))
    torch.save(global_model, get_config(key="fed_model_save"))
    del global_model
    gc.collect()
    return w_local


def test(global_model):
    global_model.eval()
    eval_acc_epoch = AverageMeter()
    global device
    load_obj = LoadDatasetEval(
        get_config(key="eval_src"),
        get_config(key="seq_length"),
        get_config(key="eval_subject"),
        get_config(key="overlap"),
        device,
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


async def send_log(message: str):
    uri = os.getenv("PPHAR_LOG_SERVER_ENDPOINT")
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    message = dt + " - [" + os.getenv("PPHAR_CORE_ID") + "] " + message
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(message)
    except Exception as e:
        print(str(e))
        pass


def write(filename, content):
    f = open("/client/keys/" + filename, "wb")
    f.write(content)
    f.close()


def read(filename):
    f = open("/client/keys/" + filename, "rb")
    return f.read()


def save(HE):
    s_context    = HE.to_bytes_context()
    s_public_key = HE.to_bytes_public_key()
    s_secret_key = HE.to_bytes_secret_key()
    s_relin_key  = HE.to_bytes_relin_key()
    s_rotate_key = HE.to_bytes_rotate_key()
    write(filename="s_context", content=s_context)
    write(filename="s_public_key", content=s_public_key)
    write(filename="s_secret_key", content=s_secret_key)
    write(filename="s_relin_key", content=s_relin_key)
    write(filename="s_rotate_key", content=s_rotate_key)


def load():
    HE = Pyfhel()
    HE.from_bytes_context(read('s_context'))
    HE.from_bytes_public_key(read('s_public_key'))
    HE.from_bytes_secret_key(read('s_secret_key'))
    HE.from_bytes_relin_key(read('s_relin_key'))
    HE.from_bytes_rotate_key(read('s_rotate_key'))
    return HE


def decrypt_global_model(HE, enc_w_avg):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    message = "Decrypting the encrypted global model.."
    print(message, flush=True)
    loop.run_until_complete(send_log(message))
    shapes = get_shapes()
    for k in enc_w_avg.keys():
        enc_w = PyCtxt(pyfhel=HE, bytestring=enc_w_avg[k].encode('cp437'))
        t = HE.decryptFrac(enc_w)
        if len(shapes) < 1:
            t = t[0:shapes[k][0]]
        else:
            m = 1
            for i in shapes[k]:
                m *= i
            t = t[0:m]
            t = t.reshape(shapes[k])
        enc_w_avg[k] = torch.tensor(t, dtype=torch.double)
    return enc_w_avg


def get_shapes():
    n_channels = get_config(key="n_channels")
    n_hidden_layers = get_config(key="n_hidden_layers")
    n_layers = get_config(key="n_layers")
    n_classes = get_config(key="n_classes")
    drop_prob = get_config(key="drop_prob")
    model = SingleLSTMEncoder(n_channels, n_hidden_layers, n_layers, n_classes, drop_prob)
    if not path.exists("init.pt"):
        init()
    model.load_state_dict(torch.load("init.pt"))
    model = model.state_dict()
    shapes = {k:[] for k in model.keys()}
    for k in model.keys():
        shapes[k] = list(model[k].shape)
    return shapes


def init():
    n_channels = get_config(key="n_channels")
    n_hidden_layers = get_config(key="n_hidden_layers")
    n_layers = get_config(key="n_layers")
    n_classes = get_config(key="n_classes")
    drop_prob = get_config(key="drop_prob")
    model = SingleLSTMEncoder(n_channels, n_hidden_layers, n_layers, n_classes, drop_prob)
    model.to(device)
    model.train()
    torch.save(model.state_dict(), "init.pt")
    del model
    gc.collect()


def encrypt_model(HE, model):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        available = psutil.virtual_memory().available/(1024**2)
        message = f"The available memory is {available} Megabytes"
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        if available > 4096:
            send_memory_queue_action(mode="RESERVE")
            break
        wait = random.randint(3,5)
        message = f"Not enough memory :( waiting {wait} seconds for our slot..."
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        time.sleep(wait)

    message = "Encrypting the local model.."
    print(message, flush=True)
    loop.run_until_complete(send_log(message))
    for k in model.keys():
        if get_device_id(torch.cuda.is_available()) == -1:
            enc_t = HE.encrypt(model[k].numpy().flatten().astype(np.double))
        else:
            enc_t = HE.encrypt(model[k].detach().cpu().numpy().flatten().astype(np.double))
        model[k] = enc_t.to_bytes()
        del enc_t
        gc.collect()
    return model


def send_encrypted_model(HE, model):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    address = os.getenv("PPHAR_SERVER_HOST")
    port = int(os.getenv("PPHAR_SERVER_PORT"))
    if int(os.getenv("PPHAR_SUBJECT_ID")) in get_config(key="he_clients"):
        send_message(address=address, port=port, HE=HE, model=encrypt_model(HE=HE, model=model))
    else:
        send_message(address=address, port=port, HE=HE, model=model)
    message = "Sent the encrypted local model to " + address
    print(message, flush=True)
    loop.run_until_complete(send_log(message))
    return message


def process_encrypted_request(request, init=False):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    HE = Pyfhel()
    if not path.exists("/client/keys/s_context"):
        message = "Initializing Pyfhel session and data..."
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        HE = Pyfhel(context_params={'scheme':'ckks', 'n':2**15, 'scale':2**30, 'qi_sizes':[60, 30, 30, 30, 60]})
        HE.keyGen()
        HE.relinKeyGen()
        HE.rotateKeyGen()
        save(HE=HE)
    else:
        message = "Loading Pyfhel session and data..."
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        HE = load()

    request = json.loads(request.get_data())
    rounds = str(request["rounds"])
    w_global = from_bytes(request["data"].encode('cp437'))
    if type(w_global) != OrderedDict:
        message = "The received global model is not an OrderedDict."
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        return None

    if not init:
        w_global = decrypt_global_model(HE=HE, enc_w_avg=w_global)

    drop_keys = ["lstm.l0.ih.weight", "lstm.l0.ih.bias", "lstm.l0.hh.weight", "lstm.l0.hh.bias"]
    n_channels = get_config(key="n_channels")
    n_hidden_layers = get_config(key="n_hidden_layers")
    n_layers = get_config(key="n_layers")
    n_classes = get_config(key="n_classes")
    drop_prob = get_config(key="drop_prob")

    if int(os.getenv("PPHAR_SUBJECT_ID")) in get_config(key="dp_sgd_clients"):
        global_model = DPLSTMEncoder(n_channels, n_hidden_layers, n_layers, n_classes, drop_prob)   
        for keys in w_global.keys():
            if keys not in drop_keys:
                global_model.state_dict()[keys] = w_global[keys]
    else:
        global_model = SingleLSTMEncoder(n_channels, n_hidden_layers, n_layers, n_classes, drop_prob)
        global_model.load_state_dict(w_global)

    del w_global
    gc.collect()
    message = "Training with the new global model"
    print(message, flush=True)
    loop.run_until_complete(send_log(message))
    w_local = train(global_model, rounds)
    del global_model
    gc.collect()
    send_encrypted_model(model=w_local, HE=HE)
    del w_local
    gc.collect()
    send_memory_queue_action(mode="FREE")
    memory_queue.clear()


def process_request(request):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    request = json.loads(request.get_data())
    rounds = str(request["rounds"])
    w_global = from_bytes(request["data"].encode('cp437'))    
    if type(w_global) != OrderedDict:
        message = "The received global model is not an OrderedDict."
        print(message, flush=True)
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
    
    del w_global
    gc.collect()
    
    message = "Training with the new global model"
    print(message, flush=True)
    loop.run_until_complete(send_log(message))
    w_local = train(global_model, rounds)
    del global_model
    gc.collect()
    send_model(model=w_local)
    del w_local
    gc.collect()


def process_memory_queue_action(request):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    request = json.loads(request.get_data())
    sender = request["sender"]
    mode = request["mode"]
    message = ""
    if mode == "FREE":
        message = f"Removed {sender} from memory queue"
        if sender in memory_queue:
            memory_queue.remove(sender)
    elif mode == "RESERVE":
        if sender not in memory_queue:
            while True:
                if len(memory_queue) < 3:
                    memory_queue.append(sender)
                    break
                message = f"Waiting 2 seconds for the list to be freed / Current list = {str(memory_queue)}"
                print(message, flush=True)
                loop.run_until_complete(send_log(message))
                time.sleep(2)    
        message = f"Added {sender} from memory queue" 
    print(message, flush=True)
    loop.run_until_complete(send_log(message))


def send_memory_queue_action(mode: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if mode == "FREE":
        if os.getenv("PPHAR_CORE_ID") in memory_queue:
            memory_queue.remove(os.getenv("PPHAR_CORE_ID"))
    elif mode == "RESERVE":
        if os.getenv("PPHAR_CORE_ID") not in memory_queue:
            memory_queue.append(os.getenv("PPHAR_CORE_ID"))

    message = f"Sending memory queue with mode = {mode} / Current list = {str(memory_queue)}"
    print(message, flush=True)
    loop.run_until_complete(send_log(message))    
    payload = {
        'mode': mode,
        'sender': os.getenv("PPHAR_CORE_ID"),
    }
    subjects = get_config("subjects")
    for s in subjects:
        address = "subject" + str(s) + ".pphar.io"
        url = "http://" + address + ":5000" + "/encrypting"
        requests.post(url, data=json.dumps(payload), timeout=None)
