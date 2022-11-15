#!/bin/bash

echo "Cleaning the mess.."$
docker stop $(docker ps -a -q  --filter ancestor=lekssays/pphar-client:gpu)

docker rm $(docker ps -a -q  --filter ancestor=lekssays/pphar-client:gpu)

docker stop $(docker ps -a -q  --filter ancestor=lekssays/pphar-client:cpu)

docker rm $(docker ps -a -q  --filter ancestor=lekssays/pphar-client:cpu)

rm ./client/init.pt

echo "Generating docker-compose.yaml..."
python3 generator.py -g $1

sleep 5

# echo "Starting servers..."
# docker-compose up -d $(<servers.txt)

# sleep 5

echo "Starting subjects..."
docker-compose up -d $(<peers.txt)

# echo "Sleeping a little bit zzz..."
# sleep 5

# echo "Initialize the model"
#curl http://0.0.0.0:8585/init
