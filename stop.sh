#!/usr/bin/env sh

docker stop $(docker ps -q -f name="parac-1") docker rm parac-1
