import subprocess
import requests
import json
import time
import os

def report(exited_containers: list):
    containers = ",".join(exited_containers)
    print(containers)
    url = "http://" + os.getenv("PPHAR_SERVER_HOST") + ":" + os.getenv("PPHAR_SERVER_PORT") + "/failed"
    payload = {
        'containers': containers,
    }
    res = requests.post(url=url, json=payload, headers={'Content-Type': 'application/json'}, timeout=None)


def get_exited_containers():
    command = "docker ps -a  --filter ancestor=lekssays/pphar-client:gpu --filter status=exited --format '{{.Names}}'"
    exited_containers = subprocess.check_output(command, shell=True).decode()
    exited_containers = exited_containers.split("\n")
    exited_containers.remove("")
    return exited_containers


def main():
    print("Starting docker status checker cronjob")
    while True:
        exited_containers = get_exited_containers()
        print("Exited containers = ", exited_containers)
        if len(exited_containers) > 0:
            report(exited_containers)
        time.sleep(5)

if __name__ == '__main__':
    main()