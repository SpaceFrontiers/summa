---
title: Deployment and proxying
parent: Extras
---

# Deployment and proxying

Use the [Summa broker](../guides/broker.md) to route gRPC requests to server
instances or partition an index. It exposes the same public search and indexing
protocol as the server, plus a broker control service.

A reverse proxy must support gRPC over HTTP/2 and preserve streaming and RPC
deadlines. Configure transport security and access control for your deployment;
the local quick-start example is not a public endpoint configuration. Summa 2
does not include the original REST or IPFS HTTP gateway.

For file-based browser search, host an immutable index snapshot on an HTTP
service that supports range requests and appropriate CORS headers, then use
[RemoteIndex](../apis/js-api.md).
