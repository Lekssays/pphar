FROM occlum/occlum:latest-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN sudo apt-get update

RUN echo "y" | sudo apt-get install python3-dev

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt