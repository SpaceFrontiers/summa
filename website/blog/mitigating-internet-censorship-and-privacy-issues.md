---
title: Mitigating Internet Censorship and Privacy Issues
parent: Blog
nav_order: 2
---

<style rel="stylesheet">
figure > img {
    display:block;
    margin-left:auto;
    margin-right:auto;
    border: 5px solid #444;
}
figure > figcaption {
    text-align: center;
    font-size: 75%;
}
</style>

# Mitigating Internet Censorship and Privacy Issues with IPFS and WASM

<i>[@PashaPodolsky](https://github.com/ppodolsky)</i>

_Originally published in 2023; technical corrections reviewed September 25, 2026. The historical implementation used Tantivy. Summa 2 uses its own Rust core; the browser-search idea remains, but the old server commands and package API are not a Summa 2 tutorial._

The purpose of this article is to propose a unique combination of modern web technologies
to address the increasing problems of website censorship and privacy breaches.

We will explore how Tantivy, WebAssembly (WASM), and the InterPlanetary File System (IPFS)
can be used to create web applications with embedded databases that can be delivered and run
directly in a user's browser without relying on a remote server.

Let's start from the model that is a root of all our issues.

## Good Old Client-Server and Its Drawbacks

In 2023, the client-server architecture is still the most widely used model on the internet.
Despite ongoing efforts to revise and improve the model, it continues to be used
with all its advantages and disadvantages.

<figure>
  <img src="https://spacefrontiers.github.io/summa/assets/client-server.drawio.png" alt="client-server-model">
  <figcaption>Client-server model</figcaption>
</figure>

In a client-server model, the server provides services to clients and may have extensive computing capacities
and record a history of interactions with all clients.

For example, a user may create an account and a server records this account and allows the user to log in later.
Another example is a search engine that collects search logs to improve its search quality in the future.

These two traits are very beneficial and allow to create very complicated and functional web services
such as social networks, search engines, mobile offices, messengers and many others. However, the client-server model
has three major drawbacks that may render its usage unsuitable in particular conditions.

Firstly, the interaction between a user and a remote server is carried out through a hostile environment,
such as wires **controlled by state agencies or telecom companies**. Encryption can protect the content of
messages, but the fact of interaction is still exposed and communication can be cut by those who have physical
access to wires or political power over telecom companies.

<figure>
  <img src="https://spacefrontiers.github.io/summa/assets/censor.drawio.png" alt="censoring">
  <figcaption>Censoring</figcaption>
</figure>

Secondly, websites often log your requests, which fundamentally **leaks your privacy** because the history of your
interactions with the site is no longer exclusively yours. Requests can reveal information to the service operator, depending on the data sent and the logging policy.

Even if the connection is secure, the company owning this server may use your data for its own purposes or accidentally
leak your data to a third party.

<figure>
  <img src="https://imgs.xkcd.com/comics/privacy_opinions.png"  width=400px height=400px alt="privacy">
  <figcaption>XKCD Privacy Opinions</figcaption>
</figure>

Technical design can reduce how much personal information a service needs to receive. Keeping query processing local is useful, but it does not by itself make the application anonymous or eliminate every source of tracking.

Thirdly, the **server may go offline**, and there is nothing you can do about it. Just imagine waking up one day without Google.

What if the client-server model were revised for increased resilience against censorship and
privacy concerns? This article explores a refreshed approach that brings websites closer to
users.

It's important to note that this approach does have its limitations and to manage expectations,
we'll only focus on web applications that:

- Don't rely on confidential algorithms/data or at least may function properly without them
- Don't require private data from other users except yours to be locally functional
- Don't require a lot of computational capacity and must be launchable by typical consumer hardware

In essence, this approach involves creating apps that generate and store private data locally and
download only a small amount of public data from the network. There will be no servers executing code,
only ones that share files with the user.

The third point also interacts with data size: browser memory, storage quotas, bandwidth and query working sets remain finite, even when the complete index is fetched lazily.

## Proposed Solution

A typical website consists of three components:

- A web interface that the user interacts with
- An API that performs useful actions
- A database that stores valuable information

<figure>
  <img src="https://spacefrontiers.github.io/summa/assets/3-part-site-arch.drawio.png" alt="3-part-site-arch">
  <figcaption>3-part site architecture</figcaption>
</figure>

What if we package all three components into a single web bundle that runs in a user's browser?

<figure>
  <img src="https://spacefrontiers.github.io/summa/assets/web-bundle.drawio.png" alt="Web bundle">
  <figcaption>Web bundle</figcaption>
</figure>

Packaging involves making all components executable within browsers, and delivering these components into browsers.
Let's examine three seemingly unrelated technologies that can help us achieve these goals.

### Tantivy

The data required for your website can be stored in different forms such as relational databases,
key-value storages, or inverted indices. Each form provides unique capabilities;
for example, relational databases are suitable for recording items such as users or actions,
key-value storages can be used for counting or caching items, and inverted indices are ideal
for text searches.

Here we are going to consider Tantivy search library, published by a former Google employee in 2017.
Tantivy is a Rust search library inspired by Lucene. Relative speed depends on the workload, version, settings and hardware. Its architecture
is described in detail in [other article](https://spacefrontiers.github.io/summa/blog/how-search-engines-work),
here I will only mention the most important properties of Tantivy for our case:

- Its compressed inverted index supports efficient local text search; comparative performance requires a matched benchmark
- Data files generated by Tantivy are **immutable**
- Query execution can be local, while the amount of index data read depends on the query and cache state

By data immutability I mean the following: the data is saved in a set of files
called a segment after, and files are not modified after commit. The next commit saves a new batch of data in new files, and
the fact of deletion of a row is stored as a bit in a bit mask next to the existing segment. Data updates
are implemented as delete and insert operations, and therefore the segment itself remains unchanged during
any operations with data. Data immutability is essential in a network environment because aggressive
caching of everything becomes possible.

Poor locality is the main problem that prevents running an arbitrary database on top of a
p2p system or network file system. Random reads overload the network and simply exhaust it with
any significant load. By combining certain approaches, Tantivy was able
to achieve high locality for all components of the search index.

In practice, locality means that not all index files need to be downloaded into the browser
for executing search queries locally, but only a portion of the relevant files.

### WASM

WASM is a byte-code format that can be executed in web browsers.
It was first introduced in 2015 and has since undergone several years of development.
Despite initial setbacks caused by the Meltdown and Spectre vulnerabilities, interest in WASM has been
growing in recent years. The main advantage of WASM is that programs compiled in this format can run in
a web environment, namely in browsers. Toolchains for compiling into WASM are available for programming languages
such as C++ and Rust what makes the development much easier.

By the end of 2022, I managed to compile Summa into a single 5MB binary and create the `summa-wasm` library,
which also provided JavaScript bindings to that version of Summa. The current `summa-wasm` supports browser search and indexing; native-only server and filesystem features are separate. Additionally, a networking layer was written to substitute range file reads
with range network requests. The original prototype could translate small range reads into network requests. Read counts and chunk sizes are workload-specific, not a bound for all queries or a statement about the current binary size.

<figure>
  <img src="https://spacefrontiers.github.io/summa/assets/web-bundle-explained.drawio.png" alt="web-bundle">
  <figcaption>Web bundle</figcaption>
</figure>

Furthermore, the `summa-wasm` library implements aggressive caching policies that dramatically reduce
the number of bytes needed to be downloaded for executing queries.

### IPFS

At the moment, we have a search engine that functions within a browser and retrieves search indices
data files through HTTP requests when executing a search query. The missing piece that would take us
into the realm of P2P is IPFS.

IPFS is a well-established technology that was introduced in 2013. Simply put, it operates similarly
to BitTorrent, enabling you to download files from "peers" using file identifiers. There is no central
authority for file hosting, so as long as you have good connectivity with the IPFS peer network and peers
are seeding the required files, you can access any files you need.

The IPFS software also provides crucial components, such as the HTTP IPFS Gateway,
which allows you to load files from IPFS using standard HTTP protocol.
The gateway opens browsers to IPFS and hence bridges the software executing within the browser with IPFS.

<figure>
  <img src="https://spacefrontiers.github.io/summa/assets/web-bundle-explained-ipfs.drawio.png" alt="web-bundle-with-ipfs">
  <figcaption>Web-bundle with IPFS</figcaption>
</figure>

The general idea is straightforward: we put web applications and data files in a single IPFS directory
and distribute the directory through IPFS. The HTTP IPFS Gateway allows us to access all files of the
bundle through the HTTP protocol, including the `index.html` that will be rendered by browsers
as usual HTML file, serving us as the entry point to our web application.

## Summa

Summa is a full-text search engine that combines
three technologies from above to help you in creation web bundles deliverable through IPFS.

A little over a year ago, I started developing the Summa server, which initially added

- a GRPC API for search and indexing to Tantivy
- the ability to index from Kafka topics
- the fasteval2 language for describing ranking functions
- and some extra search functions

At the same time Summa has been made compatible with WASM.
All Summa parts, including the network layer for loading index parts with HTTP requests,
have been shaped into `summa-wasm` module which now allows you to execute search queries
over Summa indices directly in browser.

### Why It Is So Special?

With the integration of `summa-wasm` into your web application, you will have the ability to
access data by key and perform full-text search queries on large datasets,
making it easier to build a very important class of web applications: **search engines with fewer centralized points of failure**.

Search engines encompass not only classical search but also news feeds and encyclopedias,
all of which have suffered greatly from censorship in recent years.

If your web application requires a different type of database, you have the option of using Summa
as an alternative or compiling the necessary databases into WASM, although this may require
a significant effort.

Regardless of your choice, it is crucial to re-architect your application to be more open.
The more private parts and centralized server APIs your application has, the higher the
risk of censorship through attacks on the servers or their infrastructure.

Now let's look at how the original prototype packaged the application. Decentralization can improve resilience, but gateways, peers and network access can still be blocked.

## Practice in the Original Prototype

The following walkthrough records the original 0.x deployment. For current browser APIs, use the [Summa 2 WASM guide](https://github.com/SpaceFrontiers/summa/blob/main/summa-wasm/README.md). Summa 2 does not include the old Kafka integration or a bundled HTTP IPFS gateway.

### Create a search index on the server

Basic example on how to create a search index with Summa you may find in [Quick-start guide](https://spacefrontiers.github.io/summa/quick-start).
The process is not different from any familiar to you process of database population with data.

### Create web-interface that uses this search index

At this point you must create a web-application, something like a PWA
that makes all operations locally and may interact with Summa database through `summa-wasm` bindings
and supposing that index data files lay somewhere near.

Summa provides [an example of news feed site](https://github.com/izihawa/earth-times) that may be used as a base for your application.

### Bundle your web-site

Now we should bundle all together.

<figure>
  <img src="https://spacefrontiers.github.io/summa/assets/summa-publisher.drawio.png" alt="web-bundle-with-ipfs">
  <figcaption>What summa-publisher does</figcaption>
</figure>

```bash
# Endpoint of Summa GRPC API
summa_api=0.0.0.0:8082
# Here should be your index name
index_name=ipfs_index_name
# Path to compiled web application with index.html
web_app_path=web/dist

web_cid=$(ipfs add --pin -Q -r --hash=blake3 $web_app_path)
data_cid=$(ipfs add --pin -Q -r --hash=blake3 --nocopy data/bin/$index_name)

ipfs files cp /ipfs/"$web_cid" /summa-web
ipfs files cp -p /ipfs/$data_cid /summa-web/data/$index_name

ipfs files stat --hash /summa-web
```

### Use!

Open the bundle through a local or public [IPFS gateway](https://docs.ipfs.tech/concepts/ipfs-gateway/). The original server included a gateway; Summa 2 expects a separate gateway or application-provided fetch layer.

### Optionally set IPNS or DNS name

IPFS supports [DNSLink specification](https://docs.ipfs.tech/concepts/dnslink/) for aliasing IPFS CIDs. You may
set DNS records to point to your web bundle, and it will be loaded by browsers with installed IPFS Companion extension.
Brave removed its integrated local IPFS node support in 2024; do not rely on the original browser setup. The [IPFS migration guide](https://blog.ipfs.tech/2024-brave-migration-guide/) describes alternatives using IPFS Desktop.

Afterwards, you will have a website with an embedded database that may be accessed through HTTP IPFS Gateway.
Updating the search database and the website itself will require some effort, but it is well-supported
due to two factors: index immutability and IPFS caching. All updates to the database will be stored in
a separate segment that will be distributed as part of a new bundle. Meanwhile, the IPFS daemon will not redownload
existing parts of the index, meaning only updates will be downloaded.

## Conclusion

Distributing search indices in Tantivy format provides several distinct and beneficial
properties. As previously mentioned, Tantivy boasts impressive locality, which means that only a
small portion of the remote data is needed to be downloaded to execute a local search query.

A participating IPFS node may cache and reprovide fetched blocks, depending on its configuration. A browser making HTTP requests to a gateway does not automatically become a provider: the gateway participates on its behalf. See the [IPFS privacy guidance](https://docs.ipfs.tech/how-to/privacy-best-practices/). Popular blocks can benefit from more caches, but availability still depends on providers and pinning.

This characteristic makes the entire system more self-balancing.

<figure>
  <img src="https://spacefrontiers.github.io/summa/assets/p2p-search.drawio.png" alt="p2p-search">
  <figcaption>Distribution of index parts</figcaption>
</figure>

The chunking and caching in IPFS also offer advantages, as even after updates, most of the search
index remains intact and thus, there is no need to redownload these files. Different approaches to
index layout can be used to further reduce the amount of data that needs to be transferred.

For example, in a news feed site, you could sort all your news items by their time and create new
chunks of data every hour or day. Old parts of the search index may not even be requested most of
the time, as people typically only read the latest news.

Local query execution can avoid sending the literal query to a search server. However, requested CIDs, byte ranges, timing and IP addresses can reveal information about the user's interests. IPFS is a public network, and gateway operators can observe requests; fetching index slices is not private information retrieval. See [IPFS privacy and encryption](https://docs.ipfs.tech/concepts/privacy-and-encryption/).

Once the required application and data are fully local, a query can run without network requests. This reduces that query's network exposure, provided the application also avoids analytics, remote fonts and other background requests. It is a useful property, not a general guarantee of anonymity.

Also, no surprise that services built on top of IPFS are better suited for interplanetary
communication than client-server services, as data transmission is more reliable and less
sensitive to delays compared to command transmission. Furthermore, the transmitted chunks
of data become available to all local peers on the other planet.
