---
layout: page
subtitle: Quick Start
permalink: /quick-start
---
## Table of Contents
- [Setup](#setup)
- [Fill With Documents](#fill)
- [Query](#query)

## Setup <a name="setup"></a>
Both server and client are distributed through the package systems, `Cargo` and `pip`.
Also, there is a prebuilt `summa-server` Docker image hosted on

### Summa Server
#### Docker Way
Prebuilt image: [izihawa/summa-server](https://hub.docker.com/repository/docker/izihawa/summa-server)

```bash
# Generate config for `summa-server`
docker run izihawa/summa-server generate-config -d /data \
-g 0.0.0.0:8082 -m 0.0.0.0:8084 > summa.yaml
# Create diectory for storing index
mkdir data
# Launch `summa-server`
docker run -v $(pwd)/summa.yaml:/summa.yaml -v $(pwd)/data:/data \
-p 8082:8082 -p 8084:8084 \
izihawa/summa-server serve /summa.yaml
```

#### Cargo Way
```bash
# Install through `cargo`
cargo install summa
# Generate config for `summa-server`
cargo run -r summa generate-config > summa.yaml
# Launch `summa-server`
cargo run -r serve summa.yaml
```

### Aiosumma

#### Pip way
```bash 
pip install -U aiosumma
```

## Fill With Documents <a name="fill"></a>
```bash
# Download sample dataset
wget https://dumps.wikimedia.org/other/cirrussearch/current/enwikibooks-20220523-cirrussearch-content.json.gz
# Create schema
echo "
{% include_relative files/summa-wiki-schema.yaml %}
" > schema.yaml
summa-cli localhost:8082 - create-index schema.yaml
# Upload documents 
```
## Query <a name="query"></a>
```bash

```