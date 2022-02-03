---
title: Replicable and Self-Accessible Datasets for Mitigating Internet Censorship and Privacy Issues in Practice
parent: articles
nav_order: 1
---

SENTENCES
The most prominent consequence of such approach given that is may be replicated entirely (required condition) without any effort, you are technically no owner of the site anymore.

Current methods for bypassing internet censorship usually involve using virtual private networks (VPNs) or similar technologies
to reroute requests for restricted websites through intermediate nodes that have unrestricted access to these sites.
However, what if the approach was reversed, and websites were brought through P2P technologies closer to users facing censorship?

# Intro

In 2023, the client-server architecture is still the most widely used model on the internet.
Despite ongoing efforts to revise and improve the model, it continues to be used
with its advantages and disadvantages.

![client-server](/summa/assets/client-server.drawio.png)

In a client-server model, the server provides services and may have extensive computing power
and a history of interactions for building advanced services. For example, a user may create
an account and then use it while remote server has the opportunity to bind all user interactions
together , or a search engine may collect search logs to improve its search results in the future.
However, the client-server model has two major flaws that make it increasingly unsuitable for use nowadays.

**Censoring connections** 

The interaction between a user and a remote site is carried out through an often uncontrolled
environment, such as wires controlled by state agencies or telecommunications companies.
Encryption methods can protect the content of messages, but the fact of interaction is still
exposed and can be cut by those with physical access to internet exchange nodes or political
control over telecom companies.

**Collecting of personal data** 

Every time a user sends a request to a server, they are exposing their data.
Even if their connection is secure, the target website knows that the request belongs to them.
Laws such as GDPR, CCPA and others aim to protect users from uncontrolled tracking, but they can lead
to greater inequalities if most society does not have access to private data but hackers
and state agents do. The only real way to maintain privacy is through technology.

What if the client-server model was revised for important and sensitive interactions 
with information systems, such as reading mass media and searching in the web?
Proposed approach would bring particular web-sites closer to users through P2P technologies, 
eliminating the need for requests to remote centralized servers and providing a more private 
and censorship-free experience. By considering this approach, 
we can mitigate the two flaws outlined above and improve our use of the Internet.
Though we have to restrict our approach to sites that does not require intensive interactions between 
users such as social networks.

## Overview

Bringing websites closer to users requires delivering all components, not just the frontend, 
but also the backend and data. Although the world has advanced in creating visually appealing 
interfaces, its capabilities are limited when it comes to incorporating backend computations
into the browser. Nevertheless, there are technologies that allow for this in manner comfortable 
for Internet users.

### Search engine

### IPFS

IPFS is relatively mature technology appeared in 2013. In simple words, it is like BitTorrent,
allowing you to download files from "peers" by file identifiers. As there is no central authority
for file hosting, nobody may prevent you from getting any files given you have rather good
connectivity with network of IPFS peers.

As any infrastructural software, IPFS provides an abstraction of file system that may be used by
casual users and developers. It means that having an IPFS installed you may store your files
in IPFS in directories, access files in directories created by others etc.
There is a lot of limitations in abstraction provided by IPFS, i.e. directories are immutable, no human-readable paths etc.
but basically it gives developers the most needed primitive for building software up to it.

### WASM


