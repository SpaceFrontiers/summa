---
title: Replicable and Self-Accessible Datasets for Mitigating Internet Censorship and Privacy Issues in Practice
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
Summa, built on Tantivy, has succeeded in this field, providing excellent performance and similar
capabilities.

### WASM

WASM (Web Assembly) is a format of byte-code that can be executed in web browsers.
It was first introduced in 2015 and has since undergone several years of development.
Despite initial setbacks due to the Meltdown and Spectre vulnerabilities,
interest in WASM has been growing again in recent years.
The main advantage of WASM is that it can be executed in browsers,
meaning that programs compiled in this format will run in a web environment.
For programming languages such as C++ and Rust, there are already available tools
that make compiling to WASM easier.

### IPFS

IPFS is relatively mature technology appeared in 2013. In simple words, it is like BitTorrent,
allowing you to download files from "peers" by file identifiers. As there is no central authority
for file hosting, nobody may prevent you from getting any files given you have rather good
connectivity with network of IPFS peers.

As any infrastructural software, IPFS provides an abstraction of file system that may be used by
casual users and developers. It means that having an IPFS installed you may store your files
in IPFS in directories, access files in directories created by others etc.
There is a lot of limitations in abstraction provided by IPFS, i.e. directories are immutable, 
no human-readable paths etc. but basically it gives developers 
the most needed primitive for building software up to it.




