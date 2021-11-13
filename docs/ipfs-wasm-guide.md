---
layout: page
title: IPFS Publish + WASM Browsing
permalink: /ipfs-wasm-guide
---
Current guide is a crossroad of three wonderful technologies: [IPFS](https://docs.ipfs.io/), [WASM](https://webassembly.org/getting-started/developers-guide/) and Summa.
We learn here how to: 
**create search index** on the server, 
**replicate it to IPFS**
and **open and use it from inside your browser** without the need of server.

Launching search engine in browser means all computations required for search would be done by your browser. Search engine will request data chunks over HTTP requests and that's all.
In perspective, such approach may dramatically increase privacy of search in the Internet. If your search query doesn't trip over network then nobody can count it.

Moreover, local processing of search queries would allow to use full-featured search engine on statically hosted sites or even in decentralized systems such as IPFS.

### Configuring Summa

Enable IPFS support during [config generation](/summa/quick-start) with `-i` flag that accepts IPFS API endpoint:
```bash
docker run izihawa/summa-server:testing generate-config -d /data \
-g 0.0.0.0:8082 -m 0.0.0.0:8084 -i 0.0.0.0:5001 > summa.yaml
```
or add IPFS support to existing instance by adding section
```yaml
ipfs:
  api_endpoint: "0.0.0.0:5001"
```
into your `summa.yaml`

Keep in mind that for Summa launched in Docker you should use `host.docker.internal:5001` address if IPFS is launched on host

### Creating Sample Dataset

Follow quick-start guide for [creating new index](/summa/quick-start#setup)

In the end you will have index that we will publish to IPFS and view through browser.

### Publish index to IPFS <a name="ipfs"></a>

Publishing may be done by synchronizing state between local files and remote [MFS](https://docs.ipfs.tech/concepts/file-systems/#mutable-file-system-mfs).
You need to have IPFS with write access through API. Everything other is delegated to Summa and IPFS.

```bash
# Publish index to IPFS and return keys
# Payload is extra settings required for opening and searching in the index. It is subject of changing in the nearest future.
summa-cli localhost:8082 - publish-index books --payload "{'default_fields': ['title', 'text'], 'multi_fields': [], 'name': 'books'}"

# Check if index is published to MFS
ipfs files stat /index/books
```

Here you will see hashes and sizes of published index. Copy `Hash` of the index, you will need it on the next step.

### Integrate it with browser <a name="web"></a>

Summa repository has [an example web interface](https://github.com/izihawa/summa/tree/master/summa-web-example) that is capable to use IPFS-placed search index.
You need installed `node` and `npm` for launching web-interface locally.
```bash
# Clone repository to local disk
git clone https://github.com/izihawa/summa

# Move to web interface example
cd summa/summa-web-example
```

Open `src/index.ts` and set IPFS hash that you retrieved in the previous section.

Launch time!

```bash
npm i && npm run dev
```

Now, you should open the link that appeared in your Terminal after the typing of the last command.
Type 1-2 words search query into the input box and press Enter. Search results will appear in seconds.