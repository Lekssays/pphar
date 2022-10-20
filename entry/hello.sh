#!/bin/bash
set -e

echo "Welcome from $PPHAR_CORE_ID"

mkdir /models
mkdir /models/global

if [[ "$PPHAR_CORE_ID" == *"server"* ]]; then
  # python3 /server/server.py 
  tail -f /dev/null
else
  # python3 /client/client.py 
  tail -f /dev/null
fi
