### Table of Contents
- [Setup](#setup)
- [Fill With Documents](#fill)
- [Query](#query)

In this quick-start we will create index for searching in WikiBooks. There are two essential parts, Summa server responsible for
indexing text data and Summa client that is required for communicating with Summa server. 
Although there is an [GRPC API](summa/proto) you may want to use, here we will use Summa client implemented in Python.

### Setup <a name="setup"></a>

#### Prerequisite:
- [Python3](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/)

Both server `summa-server` and Summa client (named `aiosumma`) are distributed through the package systems, `Cargo` and `pip`.
Also, there is a prebuilt `summa-server` Docker image hosted on Dockerhub that we are going to use.

#### Aiosumma
`aiosumma` is a python package that allows to use Summa GRPC API from Python and Terminal.

```bash
# (Optional) Create virtual env for `aiosumma`
python3 -m venv venv
source venv/bin/acticate

# Install aiosumma
pip3 install -U aiosumma
```

#### Summa Server
Summa server is a main guy at the party. This Rust-written software manages search indices and allows to do search queries.

```bash
# Pull actual image for `summa-server`
docker pull izihawa/summa-server:testing
```

### Fill With Documents <a name="fill"></a>
Here we download WikiBooks dumps and index them in Summa server.

```bash

# Create diectory for storing index
mkdir data

# Generate config for `summa-server`
docker run izihawa/summa-server:testing generate-config -d /data \
-g 0.0.0.0:8082 -m 0.0.0.0:8084 > summa.yaml

# Launch `summa-server`
docker run -v $(pwd)/summa.yaml:/summa.yaml -v $(pwd)/data:/data \
-p 8082:8082 -p 8084:8084 \
izihawa/summa-server:testing serve /summa.yaml
```

Then, we should open another Terminal session.

```bash
{% include download-dump-snippet.sh %}

{% include import-data-to-summa-snippet.sh %}
```

Now, we have WikiBooks databases indexed locally.

### Query <a name="query"></a>
Let's do a test query!

```bash
# Do a match query that returns top-10 documents and its total count
summa-cli localhost:8082 - search books '{"match": {"value": "astronomy"}}' '[{"top_docs": {"limit": 10}}, {"count": {}}]'
```
