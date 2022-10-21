# pphar

A Privacy-preserving Federated Learning Framework

## Getting Started

### Dependencies:

- Install docker and docker-compose.

- Install Python 3.8.10 and pip 20.0.2

- Install pip requirements:
```
pip3 install -r requirements.txt
```

### Run pphar:

- Keep peers data in `./data/` in the format `./data/:id` where `id` is the identifier of the peer (e.g., 1, 2, 3, etc.).

- Modify the config files in `./server/config.json` and `./client/config.json`.

- Start the logging server:

```
python3 ./logs/server.py 
```

- Start the simulator where `SERVERS`, `PEERS`, and `GPU` denote the number of servers, the number of peers, and the existence of a GPU (-1 for CPU and 1 for GPU).
```
./simulate.sh SERVERS PEERS GPU
```

Example of an experiment with one server two peers without GPU.
```
./simulate.sh 1 2 -1
```

- Clean the environment:
```
./clean.sh
```