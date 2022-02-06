---
title: Mitigating Internet Censorship and Privacy Issues with IPFS in Practice
parent: Articles
nav_order: 1
---

# Intro

In 2023, the client-server architecture is still the most widely used model on the internet.
Despite ongoing efforts to revise and improve the model, it continues to be used
with its advantages and disadvantages.

<figure>
  <img src="/summa/assets/client-server.drawio.png" alt="client-server-model" style="display:block; margin-left:auto; margin-right:auto">
  <figcaption style="text-align: center; font-size: 75%">Client-server model</figcaption>
</figure>

In a client-server model, the server provides services and may have extensive computing power
and keep a history of interactions. For example, a user may create
an account and server records this account, or a search engine may collect search logs to improve its search results in the future.
However, the client-server model has two major flaws that make it increasingly unsuitable for use nowadays.

The interaction between a user and a remote site is carried out through an often uncontrolled
environment, such as wires **controlled by state agencies or telecommunications companies**.
Encryption methods can protect the content of messages, but the fact of interaction is still
exposed and can be cut by those with physical access to internet exchange nodes or political
control over telecom companies.

Secondly, keeping a history is important for providing some quality of service, 
but it fundamentally **leaks your privacy** because history of your interactions with the site are no longer exclusively yours.
Every time you send a request to a server, you are exposing your data. 
Even if the connection is secure, the target website knows that the request belongs to you.

Laws such as GDPR, CCPA and others aim to protect users from uncontrolled tracking, but they can lead
to greater inequalities if most society does not have access to private data but hackers
and state agencies do. The only real way to maintain privacy is through technology.

What if the client-server model was revised for important and sensitive interactions 
with information systems, such as reading mass media and searching in the web?
Proposed approach would bring particular web-sites closer to users, namely into their browsers, 
through P2P technologies, eliminating the need for requests to remote centralized servers and 
providing a more private and censorship-free experience. By considering this approach, 
we can mitigate the two flaws outlined above and improve our use of the Internet.
Though we have to restrict our approach to sites that does not require intensive interactions between 
users such as social networks. The reasons for such limitation will become obvious after reading the article.

## Overview

Usual site consists of 3 part: web-interface that we are interacting with, 
API that does something useful and database that stores something useful.

<figure>
  <img src="/summa/assets/3-part-site-arch.drawio.png" alt="3-part-site-arch" style="display:block; margin-left:auto; margin-right:auto">
  <figcaption style="text-align: center; font-size: 75%">3-part site architecture</figcaption>
</figure>

Bringing websites into user browser requires delivering all components, not just the web-interface, 
but also the API and data. Now we need to learn about three essential parts that allows us to achieve 
the required goal.

### Search engine

The data required for your website can be stored in different forms such as relational databases,
key-value storages, or inverted indices. Each form provides unique capabilities;
for example, relational databases are suitable for recording items such as users or actions,
key-value storages can be used for counting or caching items, and inverted indices are ideal
for text searches.

This article will focus on Summa, a faster Rust-written alternative to Lucene/ES.

Summa is built on the top of Tantivy search library, published by a former Google employee in 2017.
Tantivy's architecture is similar to Lucene, but it is faster and has a smaller code base.
A little over a year ago, I started developing the Summa server, which initially added a GRPC API
for search and indexing to Tantivy, the ability to index from Kafka topics, the fasteval2 language
for describing ranking functions, and some extra search functions. The search engine is described
in detail in my earlier article, here I will only mention the most important properties of Tantivy/Summa
for distributed search: performance, data immutability, and locality of search queries: 

- The performance of the system was already faster than Lucene/ES at the time. 
- Data immutability means that after a commit, the data is saved in a set of files called a segment, 
which is not modified. The next commit saves a new batch of data in new files, and the fact of deletion 
of a row is stored as a bit in a bit mask next to the existing segment. Data updates are implemented 
as DELETE + INSERT, and therefore the segment itself remains unchanged during any operations with data
except for compaction (merging in terms of Lucene). Data immutability is essential in network environment because
it allows aggressive caching of everything.
- Locality of search queries means that requests require reading a small amount of data from disk.

High locality leads to fewer disk reads. Poor locality is the main problem that prevents simply running
an arbitrary database on top of a p2p system or network file system. Random reads overload the network
and simply exhaust it with any significant load. By combining certain approaches, Tantivy was able
to achieve high locality for all components of the search index.

### WASM

WASM (Web Assembly) is a format of byte-code that can be executed in web browsers.
It was first introduced in 2015 and has since undergone several years of development.
Despite initial setbacks due to the Meltdown and Spectre vulnerabilities,
interest in WASM has been growing again in recent years.
The main advantage of WASM is that it can be executed in browsers,
meaning that programs compiled in this format will run in a web environment.
For programming languages such as C++ and Rust, there are already available tools
that make compiling to WASM easier.

In the end of 2023 I've managed to compile Summa into a single 5MB binary and create
summa-wasm library that provides JS bindings to the Summa. It means that all search capabilities of
the index become available in the browser. Also, there has been written a networking layer allowed
to substitute range file reads that Tantivy does to range network requests. This way, every search request
that does 5-10 32KB reads from disk has been translated into 5-10 network requests that downloads 32KB of data.

Also, `summa-wasm` implements an aggressive caching policies that dramatically reduces the number of bytes
required for doing the query.

### IPFS

At present, we have a search engine that operates within a browser and retrieves search indices
through HTTP queries when executing a query. The missing component that takes us into the realm
of peer-to-peer (P2P) is InterPlanetary File System (IPFS).

IPFS is a relatively mature technology that was introduced in 2013. In simple terms, it works
like BitTorrent, allowing you to download files from "peers" using file identifiers. There is
no central authority for file hosting, so no one can prevent you from accessing any files as
long as you have good connectivity with the IPFS peer network.

IPFS provides an abstraction of the file system that is accessible to both casual users and developers,
and this abstraction is exposed through an HTTP gateway too. If IPFS is installed and a directory with files
has been "uploaded" to IPFS, the files can be accessed in the following way, provided that the HTTP
daemon is launched on port 8080.

`<picture of request to gateway>`

Well, let's put it all together:
- Create a search index on the server.
- Bundle the search index files, summa-wasm, and a web interface that uses summa-wasm to query 
the search index files.
- Start distributing the bundle through IPFS.
- Access the bundle through the HTTP IPFS gateway.

Afterwards, you will have a website with an embedded database that can be accessed through IPFS.
Updating the search database and the website itself will require some effort, but it is well-supported 
due to two factors: index immutability and IPFS caching. All updates to the database will be stored in
a separate segment that will be distributed as part of a new bundle. The IPFS daemon will not download
existing parts of the index, meaning only updates will be downloaded.
