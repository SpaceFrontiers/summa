# summa-k8s-manager

`summa-k8s-manager` runs a Summa search cluster on Kubernetes from one
`SearchCluster` custom resource. It has two parts in one process:

- **Operator**: reconciles the resource into one Deployment per shard replica,
  the broker Deployment and the broker Service.
- **Console**: a small web UI and JSON API that shows cluster status and live
  broker topology, and edits the resource after validating the change.

The resource is the source of truth. `kubectl edit searchcluster` and the
console go through the same validation in the operator; the console adds
checks against live broker state before it saves.

## Resource

`summa.spacefrontiers.org/v1alpha1`, kind `SearchCluster`, namespaced. One
`SearchCluster` per namespace: the broker discovers every pod in its namespace
that carries the shard label, so two clusters would see each other's shards.
The operator reconciles the oldest resource in a namespace and marks the others
`Invalid`.

```yaml
apiVersion: summa.spacefrontiers.org/v1alpha1
kind: SearchCluster
metadata:
  name: summa
  namespace: summa
spec:
  image: ghcr.io/spacefrontiers/summa/summa-server:2.1.3
  imagePullPolicy: IfNotPresent
  server: # defaults for every shard replica
    args: ["--max-concurrent-searches", "16"]
    env: [{ name: RUST_LOG, value: info }]
    resources: { requests: { memory: 16Gi } }
    ipcLock: true # adds CAP_IPC_LOCK for --pin-mode mlock
  shards:
    - id: "1"
      args: ["--search-threads", "8"] # appended to server.args
      resources: { requests: { memory: 64Gi } } # replaces server.resources
      replicas:
        - name: summa-server-a # Deployment name; default <cluster>-shard-<id>
          node: node-a # kubernetes.io/hostname
          dataPath: /data/summa-server
          cachePath: /data/summa-cache
  broker:
    name: summa-broker # Deployment and Service name
    image: ghcr.io/spacefrontiers/summa/summa-server:2.1.4 # default spec.image
    replicas: 1
    serviceAccountName: summa-broker
    backendMaxSearches: 16 # per backend, across all broker replicas
    args: []
    env: []
    resources: {}
  placement:
    default: single # single | reject
    rules:
      - pattern: "documents*"
        partitions: [{ shard: "1", weight: 5 }, { shard: "2", weight: 4 }]
```

### Rendered workloads

Each shard replica becomes a Deployment with one pod, `Recreate` strategy,
required node affinity on `kubernetes.io/hostname`, and `hostPath` volumes of
type `Directory` mounted at `/data` and `/cache`. The directories must exist on
the node, so a typo fails pod start instead of creating an empty index
directory. The arguments are `--addr 0.0.0.0:50051 --data-dir /data
--cache-dir /cache`, then `server.args`, then the shard's `args`. Pods carry
the broker's discovery labels: `app.kubernetes.io/name: summa-server` and
`summa.spacefrontiers.org/shard-id`. The readiness probe uses gRPC health and
liveness uses TCP, as the [broker](broker.md#discovery) requires.

The broker Deployment runs `summa-broker --discovery kubernetes --namespace
<namespace>`. Its `--placement` arguments are rendered from
`placement.rules` in order. `--backend-max-searches` is
`backendMaxSearches / broker.replicas` (rounded down): the broker limits
admission per broker process, so with several brokers each one gets a share of
the backend's search budget. A rolling update still overlaps one extra broker,
as it did before. The broker Service exposes gRPC on 50051.

The operator applies these with server-side apply under the field manager
`summa-k8s-manager`. The rendered pod templates match the hand-written manifests
they replace, so adopting existing Deployments under the same names does not
restart them. Deployments also get `app.kubernetes.io/managed-by:
summa-k8s-manager` and `summa.spacefrontiers.org/cluster: <name>` metadata labels.
The operator only deletes Deployments that carry both labels.

### Rollouts

A spec change to a shard's Deployment restarts it (`Recreate`), and the shard
serves nothing until every index has reopened. Every search of a partitioned
index fails while one of its partitions is down. The operator therefore changes
one shard at a time. Before applying a change it compares a server-side dry run
with the live Deployment's generation. While any shard Deployment is not ready,
changes to the other shards wait, and their replicas report
`<name> waits for <other> to become ready before restarting`. A change to
`spec.image` or `spec.server` therefore restarts the shards in spec order, each
after the previous one has loaded. New replicas and metadata-only changes are
not held. The broker rolls on its own (`maxUnavailable: 0`). Set `broker.image`
to upgrade the broker without restarting any shard.

The resource has no owner references to its workloads. Deleting the
`SearchCluster` leaves every shard and the broker running; remove them by
deleting the shards from the spec first.

## Validation

The operator refuses to reconcile an invalid spec. It sets
`Ready=False, reason=Invalid` with every error and its field path, and does not
change any workload. Static rules:

- Shard ids are non-empty label values without `,`, `:` or `=`, and unique.
- Every replica's Deployment name is a unique DNS label.
- Each `(node, path)` pair is used by at most one replica, across data and
  cache paths. Two summa-server processes must never share a directory.
- Paths are absolute.
- A shard has exactly one replica. summa-server has no follower replication
  yet ([broker phase 3](broker.md#phase-3-designed-not-yet-built-masterfollower-replication)):
  a second pod with the same shard id would serve a copy that never receives
  writes, and the broker refuses writes to a shard with several unlabeled
  members. The list form leaves room for followers once the server supports
  them.
- Placement patterns are non-empty and contain no `=`. Each rule lists at least
  one partition. Partitions name existing shards, at most once per rule.
  Weights are positive and only apply to rules with several partitions.
- `server.args` and shard `args` do not set flags the operator renders
  (`--addr`, `--data-dir`, `--cache-dir`). A shard does not repeat a flag that
  `server.args` already sets. Broker `args` do not set `--addr`,
  `--discovery`, `--namespace`, `--placement`, `--placement-default` or
  `--backend-max-searches`.
- `backendMaxSearches >= broker.replicas`. A shard whose args set
  `--max-concurrent-searches` below `backendMaxSearches` is refused, because
  the broker would admit more searches than the server accepts.

The broker parses the rendered placement arguments with the same parser the
operator validates with (`summa_broker::placement_rules`).

## Live safety checks

Some changes are valid in isolation but break live indexes. The operator checks
them against the running broker before it acts. The console runs the same checks
before saving, so the user sees the reason immediately.

**Placement routing.** Partition order and weights are part of a live
generation's routing contract ([partition weights](broker.md#partition-weights)).
For each index in the broker's `GetTopology`, the operator resolves its routing
under the running rules (the `--placement` arguments of the broker Deployment)
and under the proposed rules: shards in order, weights reduced by their
greatest common divisor. If the index is partitioned, or the proposed rule
would partition it, and the two routings differ, the change is refused: upserts and deletes would reach the
wrong partition, or reads would fail on a partition without the index. Moving
the rule of an unpartitioned index to another shard is allowed. It only
decides where the next `CreateIndex` lands, as in the rebuild-and-alias-swap
migration. To change the weights of a family with a live generation, first add
an exact rule that pins the generation to its current routing.

The check runs whenever the proposed rules differ from the running ones. If the
new rules would reroute a live index, or the broker cannot be queried, the
operator keeps the broker Deployment unchanged. It sets `PlacementApplied=False` with the reason and
retries every minute. Shards are still reconciled, so a rule can name a shard
added in the same edit.

**Shard removal.** Before deleting the Deployment of a shard id that is no
longer in the spec, the operator asks the broker for its backends. If any
backend of that shard lists an index, or the broker cannot answer, the shard is
kept and `ShardsPruned=False` names it. Delete or move its indexes first. A
replica whose shard id stays in the spec (for example, a rename or a move to
another node) is not subject to this check. The new replica opens whatever
`dataPath` holds, so copy the data first ([index moves](#moving-a-shard)).
Removing a Deployment never deletes the node's directories.

**Shared host paths.** A new replica Deployment is created only when no other
pod in the namespace is scheduled on its node with a `hostPath` volume at its
data or cache path. During a rename the old pod may still be terminating
(600-second grace period). The operator waits for it, because two processes
would otherwise open the same index directory.

## Status

```yaml
status:
  observedGeneration: 7
  shards:
    - id: "1"
      replicas:
        - { name: summa-server-a, node: node-a, ready: true }
  broker: { name: summa-broker, replicas: 1, readyReplicas: 1 }
  placement: ["documents*=1:5,2:4"] # rules the broker Deployment runs
  conditions:
    - { type: Ready, status: "True", reason: Reconciled }
    - { type: PlacementApplied, status: "True", reason: Applied }
    - { type: ShardsPruned, status: "True", reason: Pruned }
```

`Ready` is true when the spec is valid, every workload is applied and every
replica and broker pod is ready. Condition messages name blocked shards,
blocked indexes and unreachable services. Reconcile outcomes are counted in
`summa_k8s_manager_reconcile_total{result}`.

## Console

The console listens on `127.0.0.1:8080` by default and has no authentication of
its own. Access it with `kubectl port-forward deploy/summa-k8s-manager 8080`. Only
expose it through an authenticating proxy: anyone who can edit the resource can
create `hostPath` pods on any node.

- `GET /api/cluster`: the resource, its status, the broker's topology and
  backends, and the cluster nodes.
- `POST /api/cluster/check` with `{ "spec": … }`: static errors, live safety
  errors, and placement changes with the indexes they affect.
- `PUT /api/cluster` with `{ "resourceVersion": …, "spec": … }`: runs the
  same checks and replaces the spec. It returns `422` with the errors, or `409`
  if the resource changed since it was read.
- `POST /api/broker/refresh`: `RefreshTopology` on the broker.

The page shows shards, replicas and their node, readiness, broker health and
hosted indexes. It also shows every index with its partitions, document counts
and the share of rows each partition holds compared with its weight. Shards,
broker replicas and placement rules are edited in forms. "Review" runs the check
and shows the spec diff. "Apply" saves it.

## Moving a shard

Changing a replica's `node` or `dataPath` restarts its Deployment on that node
and directory; renaming the replica replaces the Deployment. If the directory is empty, the shard serves no indexes, and
partitioned searches fail until it is restored. To move a shard with its data:

1. Stop writers and commit, as in the broker runbook's file-copy migration.
2. Copy the index directory to the new node.
3. Change `node`, `dataPath` and `cachePath` together in one edit.
4. Verify `GetTopology` document counts in the console before resuming writers.

## Image

`summa-k8s-manager/Dockerfile` builds a separate image. The
`summa-k8s-manager image` GitHub workflow builds and smoke-tests it on pull
requests that touch the manager, the broker, the protocol or the lockfile. On
`main` it publishes `ghcr.io/spacefrontiers/summa/summa-k8s-manager` as `latest`
and `sha-<commit>`. The image runs `summa-k8s-manager run` as a non-root user;
`summa-k8s-manager crd` prints the CRD to install.

## RBAC

The manager's service account needs `get/list/watch/update/patch` on
`searchclusters` and `searchclusters/status`, `get/list/watch/create/patch/delete`
on Deployments, `get/list/watch/create/patch` on Services, `list/watch` on pods
in the namespace, and cluster-scoped `list` on nodes for the node picker. The
broker keeps its own pod-reader role.

## Not in scope

- Follower replicas (needs server replication).
- Hot reload of placement in the broker. A placement change rolls the broker
  Deployment (`maxUnavailable: 0`).
- Creating node directories. Provision them before adding a replica.
