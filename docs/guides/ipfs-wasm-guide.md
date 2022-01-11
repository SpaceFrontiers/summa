---
title: IPFS Publish + WASM Browsing
parent: Guides
nav_order: 2
---
Current guide is a crossroad of three wonderful technologies: [Iroh](https://github.com/n0-computer/iroh), [WASM](https://webassembly.org/getting-started/developers-guide/) and Summa.
We learn here how to: 
**create search index** on the server, 
**replicate it to IPFS**
and **open and use it from inside your browser** without the need of server.

Launching search engine in browser means all computations required for search would be done by your browser. Search engine will request data chunks over HTTP requests and that's all.
In perspective, such approach may dramatically increase privacy of search in the Internet. If your search query doesn't trip over network then nobody can count it.

Moreover, local processing of search queries would allow to use full-featured search engine on statically hosted sites or even in decentralized systems such as IPFS.

### Configuring Summa and index

Firstly, you should set up Summa Server with Iroh Store (enalbed by default) and create test index using [Quick-Start guide]((/summa/guides/quick-start) 

### Publish index to IPFS <a name="ipfs"></a>

For publishing index we should change its engine to IPFS. Then, Iroh P2P automatically makes it available to your IPFS peers:
```bash 
summa-cli localhost:8082 - migrate-index books books_iroh Ipfs
```
The command will return you CID of published index that you may use further for replicating or opening it through browser.
For example, you may find your index through `kubo`:
```bash
ipfs get <cid>
```

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