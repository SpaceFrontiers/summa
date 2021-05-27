---
layout: page
subtitle: Quick Start
permalink: /quick-start
---
## Table of Contents
- [Setup](#setup)
- [Fill With Documents](#fill)
- [Query](#query)

## Setup <a name="setup"></a>
Both server and client are distributed through the package systems, `Cargo` and `pip`.
Also, there is a prebuilt `summa-server` Docker image hosted on Dockerhub.

### Summa Server
#### Docker Way
Prebuilt image: [izihawa/summa-server](https://hub.docker.com/repository/docker/izihawa/summa-server)

```bash
# Generate config for `summa-server`
docker run izihawa/summa-server generate-config -d /data \
-g 0.0.0.0:8082 -m 0.0.0.0:8084 > summa.yaml

# Create diectory for storing index
mkdir data

# Launch `summa-server`
docker run -v $(pwd)/summa.yaml:/summa.yaml -v $(pwd)/data:/data \
-p 8082:8082 -p 8084:8084 \
izihawa/summa-server serve /summa.yaml
```

#### Cargo Way
```bash
# Install through `cargo`
cargo install summa

# Generate config for `summa-server`
cargo run -r summa generate-config -d /data \
-g 0.0.0.0:8082 -m 0.0.0.0:8084 > summa.yaml

# Launch `summa-server`
cargo run -r serve summa.yaml
```

### Aiosumma

#### Pip way
```bash 
pip install -U aiosumma
```

## Fill With Documents <a name="fill"></a>
```bash
# Download sample dataset
wget https://dumps.wikimedia.org/other/cirrussearch/20220523/enwikibooks-20220523-cirrussearch-content.json.gz
gunzip enwikibooks-20220523-cirrussearch-content.json.gz

# Create index schema in file
cat << EOF > schema.yaml
{% include_relative files/summa-wiki-schema.yaml %}
EOF

# Create index
summa-cli localhost:8082 - create-index-from-file schema.yaml

# Upload documents
awk 'NR%2==0' enwikibooks-20220523-cirrussearch-content.json | summa-cli localhost:8082 - index-document-stream page

# Commit index to make them searchable
summa-cli localhost:8082 - commit-index page --commit-mode Sync
```
## Query <a name="query"></a>
```bash
# Do a match query that returns top-10 documents and its total count
summa-cli localhost:8082 - search page '{"match": {"value": "astronomy"}}' '[{"top_docs": {"limit": 10}}, {"count": {}}]'
# Do a faceted search query
summa-cli localhost:8082 - search page '{"match": {"value": "astronomy"}}' '[{"facet": {"field": "category", facets: ["Book:Wikibooks Stacks/Shelves", "Cookbook"]}}]'
```