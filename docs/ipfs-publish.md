---
layout: page
title: IPFS
permalink: /ipfs-publish
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

Keep in mind that for Summa launched in Docker you should use `http://host.docker.internal:5001` address if IPFS is launched on host

### Creating Sample Dataset

Follow quick-start guide for [summa](/summa/quick-start#setup)

In the end you will have index that we will publish and view through IPFS + Web.

### Publish index to IPFS <a name="ipfs"></a>
```bash
# Publish index to IPFS and return keys
summa-cli localhost:8082 - publish-index page --copy
```

Summa supports `--no-copy` mode of IPFS. However, it requires extra efforts due to IPFS inability to `add` file outside its root.

### View it on web page <a name="web"></a>



### (Optional) Using NetworkEngine is Summa <a name="network-engine"></a>