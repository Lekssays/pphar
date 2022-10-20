#!/bin/bash


echo "Starting servers..."
screen -dmS docker-compose up $(<servers.txt)

sleep 5

echo "Starting clients..."
screen -dmS docker-compose up $(<peers.txt)

