#!/bin/bash

echo "Starting the log server..."
screen -dmS python3 ./logs/server.py 

sleep 3

echo "Generating docker-compose.yaml..."
python3 generator.py -s $1 -p $2

sleep 5

echo "Starting servers..."
screen -dmS docker-compose up $(<servers.txt)

sleep 5

echo "Starting subjects..."
screen -dmS docker-compose up $(<peers.txt)

echo "Sleeping a little bit zzz..."
sleep 20

echo "Initialize the model"
curl http://0.0.0.0:8585/init
