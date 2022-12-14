#!/usr/bin/python3
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu',
                        dest = "gpu",
                        help = "0 CPU, 1 GPU",
                        default = "0",
                        required = True)          
    return parser.parse_args()


def write(filename: str, content: str):
    writing_file = open(filename, "w")
    writing_file.write(content)
    writing_file.close()


def get_config(key: str):
    with open("config.json", "r") as f:
        config = json.load(f)
    return config[key]


def generate_peers_configs(gpu: bool) -> list:
    configs = []
    _peers = []

    if gpu:
        base_filename = "./templates/peer_gpu.yaml"
    else:
        base_filename = "./templates/peer.yaml"
    subjects = get_config(key="subjects")
    n_servers = get_config(key="n_servers")
    for subject in subjects:
        config_file = open(base_filename, "r")
        content = config_file.read()
        content = content.replace("core_id", "subject" + str(subject) + ".pphar.io")
        _peers.append("subject" + str(subject) + ".pphar.io")
        content = content.replace("subject_id", str(subject))
        config_file.close()
        configs.append(content)

    _servers = []
    if gpu:
        base_filename = "./templates/server_gpu.yaml"
    else:
        base_filename = "./templates/server.yaml"
    for server in range(1, n_servers + 1):
        config_file = open(base_filename, "r")
        content = config_file.read()
        content = content.replace("core_id", "server" + str(server) + ".pphar.io")
        _servers.append("server" + str(server) + ".pphar.io")
        config_file.close()
        configs.append(content)

    write("servers.txt", " ".join(_servers))
    write("peers.txt", " ".join(_peers))
    return configs


def generate_docker_compose(configs: list):
    main_config = ""
    base_file = open("./templates/base.yaml", "r")
    base = base_file.read()
    base_file.close()
    main_config = base + "\n"
    for config in configs:
        main_config += config + "\n"     
    write(filename="docker-compose.yaml", content=main_config)


def main():
    print("docker-compose.yaml Generator for PPHAR")
    gpu = bool(int(parse_args().gpu))
    configs = generate_peers_configs(gpu=gpu)
    generate_docker_compose(configs=configs)


if __name__ == "__main__":
    main()
