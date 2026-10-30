# summa-k8s-manager

`summa-k8s-manager` runs Summa indexes on Kubernetes. Every index is a
`SearchCluster`: its shards, its broker and its broker Service. Creating a
cluster creates the index; deleting it removes the index's workloads. Clusters
share nothing, so an index restarts, scales and upgrades on its own.

The process has two parts:

- **Operator**: reconciles each `SearchCluster` into one Deployment per
  shard, a broker Deployment and a Service, and creates the index once its
  shards serve.
- **API and console**: lists clusters with live index statistics and creates,
  edits and deletes them after validation. Admin applications use the same
  API.

## Resource

`summa.spacefrontiers.org/v1alpha1`, kind `SearchCluster`, namespaced.

```yaml
apiVersion: summa.spacefrontiers.org/v1alpha1
kind: SearchCluster
metadata:
  name: documents-20261101 # index documents_20261101
  namespace: summa
spec:
  schema: | # SDL; the operator creates the index from it
    index documents_20261101 { ... }
  shard: # every shard
    image: ghcr.io/spacefrontiers/summa/summa-server:2.2.0
    args: ["--max-concurrent-searches", "16"]
    env: [{ name: RUST_LOG, value: info }]
    resources: { requests: { memory: 64Gi } }
    ipcLock: true # CAP_IPC_LOCK for --pin-mode mlock
    tolerations: [] # e.g. dedicated data nodes
  shards: # count, order and weights are fixed once the index exists
    - { node: stc-fin-2, dataPath: /data/summa/documents_20261101, weight: 5 }
    - { node: stc-fin-2, dataPath: /data3/summa/documents_20261101, weight: 5 }
    - { node: stc-fin-2, dataPath: /data2/summa/documents_20261101, weight: 4 }
  broker:
    image: ghcr.io/spacefrontiers/summa/summa-server:2.2.0
    replicas: 1
    backendMaxSearches: 16 # per shard, across all broker replicas
```

The index name is the cluster name with `-` replaced by `_`
(`documents-20261101` serves `documents_20261101`), so index names use
lowercase letters, digits and `_`. Clients find an index's broker at
`<cluster>.<namespace>.svc:50051` from the index name alone.

`shard.image` and `broker.image` (each with an optional `imagePullPolicy`) name
the images explicitly, so a broker upgrade never restarts shards. Shard
fields: `node` (`kubernetes.io/hostname`), `dataPath`, optional `cachePath`
(default `<dataPath>-cache`), `weight` (multi-shard clusters only, default 1),
`args` and `resources` (added to and replacing the `shard` values), and
`suspended` (scale the shard to zero for maintenance).

### Rendered workloads

| Object         | Name                          | Notes                                                                                      |
| -------------- | ----------------------------- | ------------------------------------------------------------------------------------------ |
| Shard `i`      | Deployment `<cluster>-<i>`    | one pod, `Recreate`, node affinity, `hostPath` `DirectoryOrCreate` at `/data` and `/cache` |
| Broker         | Deployment `<cluster>-broker` | `RollingUpdate` with `maxUnavailable: 0`                                                   |
| Broker Service | Service `<cluster>`           | gRPC on 50051                                                                              |

Shard pods carry `summa.spacefrontiers.org/cluster: <cluster>` and
`summa.spacefrontiers.org/shard-id: <i>`. Their arguments are `--addr
0.0.0.0:50051 --data-dir /data --cache-dir /cache`, then `shard.args`, then
the shard's `args`. Readiness uses gRPC health, liveness TCP, as the
[broker](broker.md#discovery) requires.

The broker runs `summa-broker --discovery kubernetes --pod-selector
summa.spacefrontiers.org/cluster=<cluster>`, so it discovers only this
cluster's shards. Its one placement rule is the cluster's routing, for
example `documents_20261101=0:5,1:5,2:4`, with `--placement-default reject`:
the broker serves exactly its own index. `--backend-max-searches` is
`backendMaxSearches / broker.replicas`, so all broker replicas together admit
no more searches per shard than the budget.

Every object has an owner reference to its `SearchCluster`. Deleting the
cluster deletes its Deployments and Service; data directories stay on the
nodes. Server-side apply uses the field manager `summa-k8s-manager`; the
operator refuses to apply over an existing object it does not own (for
example a hand-written Service of the same name) and reports `ApplyFailed`.

Shards run with `--max-concurrent-searches` set to `backendMaxSearches`, so a
shard admits exactly what all broker replicas together may send it.

## Lifecycle

1. **Validate.** An invalid spec sets `Ready=False, reason=Invalid` with every
   error and its field, and changes nothing.
2. **Shards.** Apply each shard's Deployment. A new shard waits until no
   running pod on its node mounts its directories (for example the
   terminating pod of a replaced Deployment). Changed shards restart
   together: every shard is a partition of the one index, so any shard down
   already stops the index, and staggered restarts would only lengthen the
   outage.
3. **Broker.** Apply the broker Deployment and Service.
4. **Index.** When every shard and every broker replica runs the current
   template and is ready, ask the broker which shards host the index:
   - all: the index exists (`IndexCreated=True, reason=Exists`);
   - none, with `spec.schema`: record `IndexCreated=False, reason=Creating`
     in status, then `CreateIndex` through the broker (it sends the schema to
     every shard);
   - none, without a schema: `MissingSchema`;
   - some: `PartiallyCreated`, which needs an operator: the broker refuses
     reads and writes of a partitioned index with a missing shard.
5. **Ready** when the index exists and every shard and broker replica is
   ready. A suspended shard reports `Suspended`.

The routing (`status.routing`) is fixed from the moment the index may exist:
`IndexCreated` reason `Creating`, `PartiallyCreated` or `Exists`. Shard count,
order and weights decide which shard holds each primary key, so changing them
would send upserts and deletes to the wrong shard; repartitioning is a new
cluster (a rebuild under a new name, then an alias swap). Node, paths,
arguments, resources and suspension stay editable.

The routing lives in status, not with the data. Re-creating a deleted cluster
over its old directories must use the same shards and weights: the operator
sees an existing index and accepts the spec's routing.

### Moving a shard

Changing a shard's `node` or `dataPath` restarts it on the new node and
directory, and it opens whatever that directory holds. Stop writers, commit,
copy the shard's directory (or suspend the shard and copy), then change
`node` and `dataPath` in one edit and verify document counts before resuming
writers.

## Validation

- The name is a Service name of at most 50 characters: lowercase letters,
  digits and `-`, starting with a letter (the index name follows from it).
- At least one shard. Weights are positive and only apply to clusters with
  several shards. Paths are absolute without empty, `.` or `..` components;
  data and cache paths differ.
- No two shards use overlapping directories on one node, across all
  clusters. When clusters overlap, the older one keeps running and the newer
  one is invalid: two summa-server processes must never share a directory.
- Operator-rendered flags are refused in `args` (`--addr`, `--data-dir`,
  `--cache-dir`, `--max-concurrent-searches` for shards; `--addr`,
  `--discovery`, `--namespace`, `--pod-selector`, `--shard-label`,
  `--backend-port`, `--placement`, `--placement-default`,
  `--backend-max-searches` for the broker). A shard does not repeat a flag
  that `shard.args` sets.
- `backendMaxSearches >= broker.replicas`.
- While the routing is fixed, it must equal `status.routing`.

## Status

```yaml
status:
  observedGeneration: 3
  routing: documents_20261101=0:5,1:5,2:4
  shards:
    - { ready: true }
  broker: { readyReplicas: 1 }
  conditions:
    - { type: Ready, status: "True", reason: Reconciled }
    - { type: IndexCreated, status: "True", reason: Exists }
```

`Ready=False` reasons: `Invalid`, `ApplyFailed`, `Waiting` (a shard's
directories), `Suspended`, `Progressing`, `BrokerUnavailable`, `Creating`,
`PartiallyCreated`, `MissingSchema`. Passes are counted in
`summa_k8s_manager_reconcile_total{result}`.

## API and console

Two listeners serve the same JSON API and single-page console:

- **Console** on `127.0.0.1:8080` (`--console-addr`) for `kubectl -n summa
port-forward deploy/summa-k8s-manager 8080`. No token; requests must name
  `localhost` as their host, so a web page cannot rebind its name onto it.
- **API** on `0.0.0.0:8081` (`--api-addr`) for in-cluster clients such as
  admin, served only with `--api-token-file`; every request needs
  `Authorization: Bearer <token>`.

Anyone who can create a cluster can create `hostPath` pods on any node, so
keep the token with administrators. Requests with fields the resource does
not have are refused.

| Request                                           | Result                                                                                               |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `GET /api/clusters`                               | clusters with spec, status and live shard statistics, plus nodes                                     |
| `POST /api/clusters/check` `{ name, spec }`       | validation errors, including fixed routing of an existing index                                      |
| `POST /api/clusters` `{ name, spec }`             | creates the cluster (`409` if it exists)                                                             |
| `PUT /api/clusters/{name}` `{ generation, spec }` | replaces the spec (`409` if the spec changed since that generation)                                  |
| `DELETE /api/clusters/{name}?confirm=<name>`      | deletes the cluster; refused while it carries `summa.spacefrontiers.org/deletion-protection: "true"` |

Validation failures return `422` with `{ errors: [{ field, message }] }`.

Deletion protection in the API is advisory for `kubectl`; enforce it with a
`ValidatingAdmissionPolicy` that denies deleting a `SearchCluster` with the
annotation.

## Image and RBAC

`summa-k8s-manager/Dockerfile` builds the image; the `summa-k8s-manager image`
workflow publishes `ghcr.io/spacefrontiers/summa/summa-k8s-manager` from
`main`. `summa-k8s-manager crd` prints the CRD.

The manager needs `searchclusters` (`get/list/watch/create/update/patch/delete`
and `status` `patch`), Deployments (`get/list/watch/create/patch/delete`),
Services (`get/list/watch/create/patch`), pods (`list`) in its namespace, and
cluster-scoped `list` on nodes. Broker pods run as `broker.serviceAccountName`
(default `summa-broker`), which needs `get/list/watch` on pods.

## Not in scope

- Follower replicas of a shard (needs server replication).
- Deleting data directories. Remove them on the node after deleting a cluster.
