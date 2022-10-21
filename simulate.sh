#!/bin/bash

echo "Generating docker-compose.yaml..."
python3 generator.py -s $1 -p $2

sleep 5

echo "Starting servers..."
docker-compose up -d $(<servers.txt)

sleep 5

echo "Starting subjects..."
docker-compose up -d $(<peers.txt)

echo "Sleeping a little bit zzz..."
sleep 20

echo "Initialize the model"
curl http://0.0.0.0:8585/init
