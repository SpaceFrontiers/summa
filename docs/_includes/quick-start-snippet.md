In this quick-start we will create index for searching in WikiBooks. There are two essential parts, **Summa server** responsible for
indexing text data and **Summa client** that is required for communicating with Summa server. 

Although there is a [GRPC API](/summa/apis/grpc-api) you may want to use, here we will use Summa client implemented in Python.

### Setup <a name="setup"></a>

#### Prerequisite:
- [Python3](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/)

Both server `summa-server` and Summa client (named `aiosumma`) are distributed through the package systems, `Cargo` and `pip` correspondingly.
Also, there is a prebuilt `summa-server` Docker image hosted on Dockerhub that we are going to use.

#### Summa Server
`summa-server` is a main guy at the party. This tool manages search indices and allows to do search queries. 
We are going to pull and launch `summa-server` through Docker. Pulling can be done by `docker pull`

```bash
# Pull actual image for `summa-server`
docker pull izihawa/summa-server:testing

# Create local directory for storing index
mkdir data

# Generate config for `summa-server`
# -g flag is for setting listen address of GRPC API
# -m flag is for setting listen address of Metrics API
# add -i flag for setting IPFS endpoint
docker run izihawa/summa-server:testing generate-config -d /data \
-g 0.0.0.0:8082 -m 0.0.0.0:8084 > summa.yaml

# Launch `summa-server`
docker run -v $(pwd)/summa.yaml:/summa.yaml -v $(pwd)/data:/data \
-p 8082:8082 -p 8084:8084 \
izihawa/summa-server:testing serve /summa.yaml
```

After the last command you should see starting logs of `summa-server`, something like
```bash
2022-11-17T16:14:00.712450Z  INFO main lifecycle: summa_server::servers::metrics: action="binded" endpoint="0.0.0.0:8084"
2022-11-17T16:14:00.714536Z  INFO main lifecycle: summa_server::servers::grpc: action="binded" endpoint="0.0.0.0:8082"
2022-11-17T16:14:00.752511Z  INFO main summa_server::services::index_service: action="index_holders" index_holders={}
```

#### Aiosumma
`aiosumma` is a python package that allows to use Summa GRPC API from Python and Terminal. Let's install it:
Previous Terminal session is occupied by `summa-server`. Leave it alone, open another bash terminal:

```bash
# (Optional) Create virtual env for `aiosumma`
python3 -m venv venv
source venv/bin/acticate

# Install aiosumma
pip3 install -U aiosumma
```
Now we have `summa-cli` tool for doing queries to `summa-server`

### Fill With Documents <a name="fill"></a>
Here we download WikiBooks dumps and index them in Summa server.

```bash
{% include download-dump-snippet.sh %}

{% include import-data-to-summa-snippet.sh %}
```

Now, we have WikiBooks database indexed locally.

### Query <a name="query"></a>
Let's do a test query!

```bash
# Do a match query that returns top-10 documents and its total count
summa-cli localhost:8082 - search books '{"match": {"value": "astronomy"}}' '[{"top_docs": {"limit": 10}}, {"count": {}}]'
```
