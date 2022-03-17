#!/bin/bash

cargo build --profile release -p summa-server
cp target/release/summa-server-bin summa-server/
docker build -t izihawa/summa-server:testing summa-server
docker tag izihawa/summa-server:testing izihawa/summa-server:0.14.2
docker push izihawa/summa-server:testing
docker push izihawa/summa-server:0.14.2
rm summa-server/summa-server-bin