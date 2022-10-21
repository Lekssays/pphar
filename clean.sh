docker stop $(docker ps -a -q  --filter ancestor=lekssays/pphar-client:latest)

docker rm $(docker ps -a -q  --filter ancestor=lekssays/pphar-client:latest)