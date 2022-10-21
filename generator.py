#!/usr/bin/python3
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--participants',
                        dest = "participants",
                        help = "Number of participants",
                        default = "6",
                        required = True)
    parser.add_argument('-s', '--servers',
                        dest = "servers",
                        help = "Number of servers",
                        default = "1",
                        required = True)
    parser.add_argument('-g', '--gpu',
                        dest = "gpu",
                        help = "-1 CPU, 1 GPU",
                        default = "-1",
                        required = True)          
    return parser.parse_args()


def write(filename: str, content: str):
    writing_file = open(filename, "w")
    writing_file.write(content)
    writing_file.close()


def generate_peers_configs_cpu(participants: int, servers: int) -> list:
    configs = []
    _peers = []
    base_filename = "./templates/peer.yaml"
    for participant in range(1, participants + 1):
        config_file = open(base_filename, "r")
        content = config_file.read()
        content = content.replace("core_id", "subject" + str(participant) + ".pphar.io")
        _peers.append("subject" + str(participant) + ".pphar.io")
        content = content.replace("subject_id", str(participant))
        config_file.close()
        configs.append(content)

    _servers = []
    base_filename = "./templates/server.yaml"
    for server in range(1, servers + 1):
        config_file = open(base_filename, "r")
        content = config_file.read()
        content = content.replace("core_id", "server" + str(server) + ".pphar.io")
        _servers.append("server" + str(server) + ".pphar.io")
        config_file.close()
        configs.append(content)

    write("servers.txt", " ".join(_servers))
    write("peers.txt", " ".join(_peers))
    return configs


def generate_peers_configs_gpu(participants: int, servers: int) -> list:
    configs = []
    _peers = []
    base_filename = "./templates/peer_gpu.yaml"
    for participant in range(1, participants + 1):
        config_file = open(base_filename, "r")
        content = config_file.read()
        content = content.replace("core_id", "subject" + str(participant) + ".pphar.io")
        _peers.append("subject" + str(participant) + ".pphar.io")
        content = content.replace("subject_id", str(participant))
        config_file.close()
        configs.append(content)

    _servers = []
    base_filename = "./templates/server_gpu.yaml"
    for server in range(1, servers + 1):
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
    participants = int(parse_args().participants)
    servers = int(parse_args().servers)
    gpu = int(parse_args().gpu)
    configs = None
    if gpu == -1:
        configs = generate_peers_configs_cpu(participants=participants, servers=servers)
    elif gpu == 1:
        configs = generate_peers_configs_gpu(participants=participants, servers=servers)
    generate_docker_compose(configs=configs)


if __name__ == "__main__":
    main()