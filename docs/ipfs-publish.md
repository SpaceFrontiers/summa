---
layout: page
title: IPFS
permalink: /ipfs-replication
---
## IPFS API

First of all, IPFS and Summa must be configured

### IPFS
```bash
# Enable FileStore in IPFS
ipfs config --bool Experimental.FilestoreEnabled true
```

### Summa
Enable IPFS Support during config generation
```bash
docker run izihawa/summa-server:testing generate-config -d /data \
-g 0.0.0.0:8082 -m 0.0.0.0:8084 -i 0.0.0.0:5001 > summa.yaml
```
or add it for existing instance via adding section
```yaml
ipfs:
  api_endpoint: "0.0.0.0:5001"
```
into `config.yaml`

Keep in mind that for Summa launched in Docker you should use `host.docker.internal:5001` address if IPFS is launched on host

### Publish index to IPFS <a name="ipfs"></a>
```bash
# Publish index to IPFS and return keys
summa-cli localhost:8082 - publish-index page --copy
```

Summa supports `--no-copy` mode of IPFS but it requires extra efforts due to IPFS inability to `add` file that is outside of its root.
