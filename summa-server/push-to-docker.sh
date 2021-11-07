#!/bin/bash

cargo build --profile release -p summa-server
docker build -t izihawa/summa-server:testing -f summa-server/Dockerfile .
docker push izihawa/summa-server:testing