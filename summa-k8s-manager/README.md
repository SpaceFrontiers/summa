# summa-k8s-manager

Kubernetes operator, API and console for Summa indexes. Each index is a
`SearchCluster`: its shards (one Deployment each, pinned to a node and host
directory), its broker and the broker Service clients reach it at. Creating a
cluster creates the index; deleting it removes the workloads.

- The operator waits for a replaced shard's pod to release its directories,
  creates the index once every shard and the broker serve, and keeps the
  routing (shard count, order and weights) fixed once the index may exist.
- The API (`/api/clusters`) and console list clusters with live document
  counts and create, edit and delete them after the same validation. The
  console listens on loopback for `kubectl port-forward
deploy/summa-k8s-manager 8080`; in-cluster clients use the API listener
  with the bearer token from `--api-token-file`.

```bash
summa-k8s-manager crd | kubectl apply -f -   # install the CRD
summa-k8s-manager run --namespace summa      # operator, API and console
```

Design: [docs/summa-k8s-manager.md](../docs/summa-k8s-manager.md).
