---
title: IPFS Publish + WASM Browsing
parent: Guides
nav_order: 2
---
This guide covers three powerful technologies: [Iroh](https://github.com/n0-computer/iroh), [WASM](https://webassembly.org/getting-started/developers-guide/), and Summa. In this guide, you will learn how to:

Create a search index on the server
Replicate it to IPFS
Open and use it from inside your browser without the need for a server
Running a search engine in the browser means all computations required for the search will be done by your browser. The search engine will request data chunks over HTTP requests, and that's it. In the long run, this approach could significantly increase the privacy of search on the internet. If your search query doesn't leave your network, then nobody can track it.

Additionally, local processing of search queries would allow for full-featured search engines on statically hosted sites or even in decentralized systems such as IPFS.

### Configuring Summa and the index
First, you should set up Summa Server with Iroh Store (enabled by default) and create a test index using our [Quick-Start guide](/summa/guides/quick-start) 

### Publish the index to IPFS <a name="ipfs"></a>
To publish the index, we need to change its engine to IPFS. Then, Iroh P2P will automatically make it available to your IPFS peers:

```bash 
summa-cli localhost:8082 - migrate-index books books_iroh Ipfs
```
The command will return the CID of the published index that you can use later for replicating or opening it through the browser.
For example, you can find your index using `kubo`:

```bash
ipfs get <cid>
```

### Integrate it with browsers <a name="web"></a>
The Summa repository has [an example web interface](https://github.com/izihawa/summa/tree/master/summa-web-example) that is capable of using an IPFS-hosted search index.
You need to have `node and `npm installed to launch the web interface locally.
```bash
# Clone repository to local disk
git clone https://github.com/izihawa/summa

# Move to web interface example
cd summa/summa-web-example
```

Open `src/index.ts` and set the IPFS hash that you retrieved in the previous section.

It's launch time!

```bash
npm i && npm run dev
```

Now, open the link that appeared in your Terminal after typing the last command.
Type a 1-2 word search query into the input box and press Enter. Search results will appear in seconds.