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

### Docker Way
Prebuilt image: [izihawa/summa-server](https://hub.docker.com/repository/docker/izihawa/summa-server)

```bash
# Generate config for `summa-server`
docker run izihawa/summa-server generate-config -d /data \
-g 0.0.0.0:8082 -m 0.0.0.0:8084 > summa.yaml
# Launch with mounting

```

### Native

#### Configure Cargo

## Fill With Documents <a name="fill"></a>
## Query <a name="query"></a>