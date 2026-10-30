# summa-k8s-manager

Kubernetes operator and web console for a Summa cluster described by one
`SearchCluster` resource: the shards and the node and host directories of each
replica, the broker replicas, and the placement rules.

- The operator reconciles the resource into one Deployment per shard replica,
  the broker Deployment and the broker Service. It adopts existing
  hand-written Deployments of the same names without restarting them.
- Before rolling a placement change into the broker, it checks the broker's
  live topology and refuses changes that would reroute a live index. It keeps
  a removed shard's Deployment while the broker shows the shard hosts indexes.
- The console (`kubectl port-forward deploy/summa-k8s-manager 8080`) shows shards,
  broker health, indexes and partition balance, and edits the resource after
  the same checks.

```bash
summa-k8s-manager crd | kubectl apply -f -          # install the CRD
summa-k8s-manager run --namespace summa             # operator + console
```

Design and safety rules: [docs/summa-k8s-manager.md](../docs/summa-k8s-manager.md).
