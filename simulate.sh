#!/bin/bash

echo "Cleaning the mess.."$
docker stop $(docker ps -a -q  --filter ancestor=lekssays/pphar-client:gpu)

docker rm $(docker ps -a -q  --filter ancestor=lekssays/pphar-client:gpu)

docker stop $(docker ps -a -q  --filter ancestor=lekssays/pphar-client:cpu)

docker rm $(docker ps -a -q  --filter ancestor=lekssays/pphar-client:cpu)

rm ./client/init.pt


if [[ $# -lt 1 ]] ; then
  printHelp
  exit 0
else
  MODE=$1
  LOCAL=$2
  shift
fi

if [ "${MODE}" == "init" ]; then
    echo "Initialize the model"
    curl http://0.0.0.0:8585/init
    exit 0
fi

echo "Generating docker-compose.yaml..."
python3 generator.py -g $MODE
sleep 5

if [ "${LOCAL}" == "1" ]; then
    echo "Starting servers..."
    docker-compose up -d $(<servers.txt)
elif [ "${LOCAL}" == "2" ]; then
    echo "Starting subjects..."
    docker-compose up -d $(<peers.txt)
else
    echo "Starting servers AND subjects"
    docker-compose up -d $(<servers.txt)
    sleep 2
    docker-compose up -d $(<peers.txt)
fi


function printHelp() {
  echo "Usage: "
  echo "  simulate.sh <Mode> <Local>"
  echo "    Modes:"
  echo "      "$'\e[0;32m'1$'\e[0m' - GPU
  echo "      "$'\e[0;32m'0$'\e[0m' - CPU
  echo "    Local:"
  echo "      "$'\e[0;32m'1$'\e[0m' - Running the aggregating server
  echo "      "$'\e[0;32m'2$'\e[0m' - Running the subjects
}