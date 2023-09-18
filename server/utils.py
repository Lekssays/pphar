import asyncio
import copy
import io
import json
import torch
import requests
import os
import websockets
import psutil
import time
import random
import gc
import glob

import numpy as np

from collections import OrderedDict
from datetime import datetime
from src.main_fed import FedAvg
from Pyfhel import Pyfhel, PyCtxt, PyPtxt
from multiprocessing import Process
from src.SingleLSTM import SingleLSTMEncoder
from helper import get_device_id


device_id = -1
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
    if get_device_id(torch.cuda.is_available()) == -1:
        loaded_content = torch.load(buff, map_location=torch.device('cpu'))
    else:
        loaded_content = torch.load(buff)
    return loaded_content


def send_message(address: str, port: int, data: bytes, init: bool, encrypted: bool, resumed: bool):
    global rounds
    if resumed:
        rounds = load_rounds()
    if init and rounds == 0:
        url = "http://" + address + ":" + str(port) + "/init"
    else:
        if encrypted:
            url = "http://" + address + ":" + str(port) + "/enc_models"
        else:
            url = "http://" + address + ":" + str(port) + "/models"
    payload = {
        "data": data.decode("cp437"),
        "rounds": rounds,
    }
    print(f"Model sent to {url}")
    return requests.post(url=url, data=json.dumps(payload), timeout=None)


def send_global_model(model, init=False, encrypted=False, failed=False, subjects=[], resumed=False):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    global rounds
    if resumed:
        rounds = load_rounds()

    if not failed:
        subjects = get_config(key="subjects")
    else:
        rounds -= 1
    

    data = to_bytes(content=model)
    #Here we have to send a copy of the model the TEE server
    for s in subjects:
        if get_config("local"):
            address = "subject" + str(s) + ".pphar.io"
            port = 5000
        else:
            address = os.getenv("PPHAR_REMOTE_TRAINING_SERVER")
            port = int("444" + str(s))
        if encrypted:
            message = "Sending the encrypted global model to " + address + ":" + str(port) +" / " + str(rounds)
            # message_tee = "Sending the encrypted global model to " + tee_address + ":" + str(tee_port) +" / " + str(rounds)
        else:   
            if init and rounds == 0:
                message = "Sending the initial global model to " + address + ":" + str(port) +" / " + str(rounds)
            else:
                message = "Sending the global model to " + address + ":" + str(port) +" / " + str(rounds)
                # message_tee = "Sending the global model to" + tee_address + ":" + str(tee_port) +" / " + str(rounds)

        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        p = Process(target=send_message, args=(address, port, data, init, encrypted, resumed, ))
        p.start()
        time.sleep(random.randint(0, 5))
    if init and rounds == 0:
        tee_address = "tee.pphar.io"
        tee_port = 5555
        message_tee = "Sending the initial global model to " + tee_address + ":" + str(tee_port) +" / " + str(rounds)
        print(message_tee, flush=True)
        loop.run_until_complete(send_log(message_tee))
        tee_process = Process(target=send_message, args=(tee_address, tee_port, data, init, encrypted, resumed, ))
        tee_process.start()
    rounds += 1


async def send_log(message: str):
    uri = os.getenv("PPHAR_LOG_SERVER_ENDPOINT")
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    message = dt + " - [" + os.getenv("PPHAR_CORE_ID") + "] " + message
    async with websockets.connect(uri) as websocket:
        await websocket.send(message)


def get_rounds():
    global rounds
    return rounds


def save_rounds():
    global rounds
    data = {
        "rounds": rounds,
    }
    with open('rounds.json', 'w') as f:
        json.dump(data, f)


def load_rounds():
    with open("rounds.json", "r") as f:
        data = json.load(f)
    return data["rounds"]


def EncFedAvg(HE, enc_w):
    n = 1.0 / len(enc_w)
    enc_w_avg = enc_w[0]

    for k in enc_w_avg.keys():
        for i in range(0, len(enc_w)):
            if type(enc_w[i][k]) == PyPtxt:
                enc_w[i][k] = np.double(enc_w[i][k].decode())
            if type(enc_w_avg[k]) == PyPtxt:
                enc_w_avg[k] = np.double(enc_w_avg[k].decode())
            if type(enc_w_avg[k]) == float:
                enc_w_avg[k] = np.double(enc_w_avg[k])
            if type(enc_w[i][k]) == float:
                enc_w[i][k] = np.double(enc_w[i][k])

    for k in enc_w_avg.keys():
        for i in range(1, len(enc_w)):
            enc_w_avg[k] = enc_w[i][k] + enc_w_avg[k]
        enc_w_avg[k] *= n
        HE.relinKeyGen()
        HE.rescale_to_next(enc_w_avg[k])
        enc_w_avg[k] = enc_w_avg[k].to_bytes().decode('cp437')

    return enc_w_avg


def process_request(request):
    global w_locals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    request = json.loads(request.get_data())
    model = from_bytes(content=request['data'].encode('cp437'))
    sender = request['sender']
    
    rounds = get_rounds()
    if rounds > get_config(key="epochs")  - 1:
        message = "Training finished :)"
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        return "Training finished."

    # add_subject(sender)
    w_locals.append(model)
    del model
    gc.collect()
    message = f"Received a local model from {sender}"
    print(message, flush=True)
    loop.run_until_complete(send_log(message))

    #Aggregation logic will change as number of clients will reduce.
    if len(w_locals) == (len(get_config(key="subjects"))-get_config(key="n_tee_clients")+1): #glob.glob('./subjects/*.sbj'))
        message = "Aggregating local models."
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        w_global = FedAvg(w_locals)
        w_locals.clear()

        message = "Saving the aggregated global model locally."
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        torch.save(w_global, "w_global.pt")
        

        send_global_model(model=w_global, init=False, encrypted=False)
        save_rounds()
        del w_global
        gc.collect()

        # clear_subjects_directory()

        message = f"Sent aggregated global model to all clients."
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
    
    return message


def add_subject(subject: str):
    f = open("./subjects/" + subject + ".sbj", "w")
    f.write("added")
    f.close()


def clear_subjects_directory():
    files = glob.glob('./subjects/*')
    for f in files:
        if os.path.isfile(f):
            os.remove(f)


def process_encrypted_request(request):
    global w_locals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    rounds = get_rounds()
    if rounds > get_config(key="epochs") - 1:
        message = "Training finished :)"
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        return "Training finished."

    request = json.loads(request.get_data())

    HE = Pyfhel()
    HE.from_bytes_context(request['context'].encode('cp437'))
    HE.from_bytes_public_key(request['pk'].encode('cp437'))
    HE.from_bytes_relin_key(request['rlk'].encode('cp437'))
    HE.from_bytes_rotate_key(request['rtk'].encode('cp437'))

    enc_model = from_bytes(content=request['data'].encode('cp437'))
    sender = request['sender']
    print(f"Received HE={HE} from {sender}")
    
    for k in enc_model.keys():
        if get_subject_id(name=sender) in get_config(key="he_clients"):
            enc_model[k] = PyCtxt(pyfhel=HE, bytestring=enc_model[k])
        else:
            enc_model[k] = HE.encodeFrac(enc_model[k].numpy().flatten().astype(np.double))

    add_subject(sender)
    w_locals.append(enc_model)
    del enc_model
    gc.collect()
    message = f"Received an encrypted local model from {sender}"
    print(message, flush=True)
    loop.run_until_complete(send_log(message))

    if len(glob.glob('./subjects/*.sbj')) == len(get_config(key="subjects")):
        message = f"Aggregating encrypted local models from {sender}"
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        enc_w_global = EncFedAvg(HE=HE, enc_w=w_locals)
        w_locals.clear()
        
        message = "Saving the encypted aggregated global model locally."
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
        torch.save(enc_w_global, "enc_w_global.pt")
        save_rounds()

        send_global_model(model=enc_w_global, init=False, encrypted=True)
        del enc_w_global
        gc.collect()

        clear_subjects_directory()

        message = f"Sent aggregated global model to {sender}"
        print(message, flush=True)
        loop.run_until_complete(send_log(message))
    
    return message


def get_subject_id(name: str):
    cid = ""
    for c in name:
        if c.isdigit():
            cid += c
    return int(cid)


def process_failed_request(request):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    exited_containers = request.json.get('containers')
    exited_containers = exited_containers.split(",")

    w_global = None
    if get_config("encrypted"):
        w_global = torch.load("enc_w_global.pt")
    else:
        w_global = torch.load("w_global.pt")

    cids = []
    for container in exited_containers:
        if ".pphar.io" in container:
            cids.append(get_subject_id(container))

    message = f"Sending the global model to the exited containers..."
    print(message, flush=True)
    loop.run_until_complete(send_log(message))
    send_global_model(model=w_global, encrypted=get_config("encrypted"), failed=True, subjects=cids)

    return message


def process_resume_request(request):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    message = f"Loading the global model..."
    print(message, flush=True)
    loop.run_until_complete(send_log(message))

    w_global = None
    if get_config("encrypted"):
        w_global = torch.load("enc_w_global.pt")
    else:
        w_global = torch.load("w_global.pt")

    message = f"Sending the global model..."
    print(message, flush=True)
    loop.run_until_complete(send_log(message))
    send_global_model(model=w_global, encrypted=get_config("encrypted"), resumed=True)

    return message
