#!/bin/bash
set -e

echo "Welcome from $PPHAR_CORE_ID"

export PATH="/server/miniconda/bin:$PATH"

# if [[ "${PPHAR_CORE_ID:0:6}" == "server" ]]
# then
#     cd /root/demos/hello_c
#     make
#     ./hello_world
#     cd /server
#     eval "$(conda shell.bash hook)"
#     conda activate /server/python-occlum
# else
#     python3 /client/client.py
# fi

# To keep the container running for testing purposes
tail -f /dev/null
