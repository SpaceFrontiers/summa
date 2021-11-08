#!/bin/bash

cargo build --profile release -p summa-server
cp target/release/summa-server-bin summa-server/
docker build -t izihawa/summa-server:testing summa-server
docker push izihawa/summa-server:testing
rm summa-server/summa-server-bin