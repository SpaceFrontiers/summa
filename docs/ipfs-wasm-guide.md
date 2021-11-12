---
layout: page
title: IPFS Publish + WASM Browsing
permalink: /ipfs-wasm-guide
---
Here is a crossroad of three wonderful technologies: [IPFS](https://docs.ipfs.io/), [WASM](https://webassembly.org/getting-started/developers-guide/) and Summa.
In the following guide we learn how to: 
create search index on the server, 
replicate it to IPFS 
and how to open it from inside your browser, using library allowing to fetch index directly to your browser from IPFS

### Configuring Summa

Enable IPFS support during [config generation] with `-i` flag that accepts IPFS API endpoint:
```bash
docker run izihawa/summa-server:testing generate-config -d /data \
-g 0.0.0.0:8082 -m 0.0.0.0:8084 -i 0.0.0.0:5001 > summa.yaml
```
or add IPFS support to existing instance by adding section
```yaml
ipfs:
  api_endpoint: "0.0.0.0:5001"
```
into your `config.yaml`

Keep in mind that for Summa launched in Docker you should use `host.docker.internal:5001` address if IPFS is launched on host

### Creating Sample Dataset

Follow quick-start guide for [creating new index](/summa/quick-start#setup)

In the end you will have index that we will publish and view through IPFS + Web.

### Publish index to IPFS <a name="ipfs"></a>

Publishing may be done by synchronizing state between local files and remote [MFS](https://docs.ipfs.tech/concepts/file-systems/#mutable-file-system-mfs).
You need to have IPFS with write access through API. Everything other is delegated to Summa and IPFS.

```bash
# Publish index to IPFS and return keys
# Payload is extra settings required for opening and searching in the index. It is subject of changing in the nearest future.
summa-cli localhost:8082 - publish-index books --payload "{'default_fields': ['title', 'text'], 'multi_fields': [], 'name': 'books'}"

# Check if index is published to MFS
ipfs files stats /index/books
```

Here you will see hashes and sizes of published index. Copy `Hash` of the index, you will need it on the next step.

### View it on web page <a name="web"></a>

Summa repository has [an example web interface](https://github.com/izihawa/summa/tree/master/summa-web-example) that is capable to use IPFS-placed search index.
```bash
# Clone repository to local disk
git clone https://github.com/izihawa/summa

# Move to web interface example
cd summa/summa-web-example

# Install and launch example
npm i && npm run dev
```

Now, you should open the link that appeared in your Terminal after the typing of the last command.
Type 1-2 words search query into the input box and press Enter. Search results will appear in seconds.