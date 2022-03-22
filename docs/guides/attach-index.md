---
title: Attach Summa Indices
parent: Guides
nav_order: 2
---

You can not just create indices but install downloaded indices or even access remote indices through network.
First, you should set up Summa Server with Iroh Store (enabled by default) and create a test index using our [Quick-Start guide](/summa/quick-start)

### Local File Indices

Put downloaded directory with index files to `data/bin/<index_name>` folder and then do
`summa-cli 0.0.0.0:82 attach-index <index_name> '{"file": {}}'`

### Remote File Indices
You may also attach any index available through HTTP:

