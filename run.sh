#!/usr/bin/env sh

docker-compose -f docker-compose.yaml build
docker-compose -f docker-compose.yaml up development -d

echo "Connectiong to container app-server ..."
docker exec -it parac-1 /bin/bash
