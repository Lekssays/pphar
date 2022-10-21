#!/bin/bash
set -e

echo "Welcome from $PPHAR_CORE_ID"

MODELS="/models/"
if [ -d "$MODELS" ]; then
  echo "Skipping $MODELS directory making.."
else
  echo "Creating $MODELS directory.."
  mkdir $MODELS
  GLOBAL="/models/global/"
  if [ -d "$GLOBAL" ]; then
    echo "Skipping $GLOBAL directory making.."
  else
    echo "Creating $GLOBAL directory.."
    mkdir $GLOBAL
  fi
fi

if [[ "$PPHAR_CORE_ID" == *"server"* ]]; then
  python3 /server/server.py 
  tail -f /dev/null
else
  python3 /client/client.py 
  tail -f /dev/null
fi
