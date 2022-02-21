In this quick-start we will create index for searching in WikiBooks. There are two essential parts, **Summa server** responsible for
indexing text data and **Summa client** that is required for communicating with Summa server. 

Although there is a [GRPC API](/summa/apis/grpc-api) you may want to use through tools like `grpcurl`, here we will use Summa client implemented in Python.

### Install <a name="setup"></a>

#### Prerequisite:
- [Python3](https://www.python.org/downloads/) or [grpcurl](https://github.com/fullstorydev/grpcurl)
- [Docker](https://www.docker.com/)

`summa-server` is distributed as a prebuilt Docker image hosted on Dockerhub, or may be build from sources. Summa exposes its APIs through GRPC what 
makes available it to use in all languages having GRPC client libraries. Additionally, there is an `aiosumma` Python packages
that provides Python client and CLI.

#### Summa Server
We are going to pull and launch `summa-server` through Docker. Pulling can be done by `docker pull`

```bash
# Pull actual image for `summa-server`
docker pull izihawa/summa-server:testing

# Create local directory for storing index
mkdir data

# Generate config for `summa-server`
# -a flag is for setting listen address of GRPC API
# -i flag is for setting listen address of Iroh Gateway HTTP
docker run izihawa/summa-server:testing generate-config -d /data \
-a 0.0.0.0:8082 -i 0.0.0.0:8080 > summa.yaml

# Launch `summa-server`
docker run -v $(pwd)/summa.yaml:/summa.yaml -v $(pwd)/data:/data \
-p 8082:8082 -p 8080:8080 -p 4444:4444 -p 4445:4445 \
izihawa/summa-server:testing serve /summa.yaml
```

After the last command you should see starting logs of `summa-server`, something like
```bash
2022-11-17T16:14:00.712450Z  INFO main lifecycle: summa_server::servers::metrics: action="binded" endpoint="0.0.0.0:8084"
2022-11-17T16:14:00.714536Z  INFO main lifecycle: summa_server::servers::grpc: action="binded" endpoint="0.0.0.0:8082"
2022-11-17T16:14:00.752511Z  INFO main summa_server::services::index_service: action="index_holders" index_holders={}
```

#### Aiosumma
`aiosumma` is a Python package for using Summa GRPC API from Python and Terminal. Let's install it:

```bash
# (Optional) Create virtual env for `aiosumma`
python3 -m venv venv
source venv/bin/acticate

# Install aiosumma
pip3 install -U aiosumma
```

#### grpcurl

You may also use `curl`-alike tool for reaching `summa-server` though Terminal.
You may download its binary from [their repository](https://github.com/fullstorydev/grpcurl/releases) or install through
brew on MacOS: `brew install grpcurl`

### Create Index
Summa is a schemaful search engines. It requires from you to define fields what you are going to use. Let's create
a schema for WikiBooks:

```bash
# Create index schema in file
cat << EOF > schema.yaml
{% include summa-wiki-schema.yaml %}
EOF
```

### Add Documents <a name="fill"></a>
WikiBooks provides weekly dumps of their books' database. 
Let's download their dump and index it in Summa:

```bash
{% include download-dump-snippet.sh %}
```

```bash   
{% include import-data-to-summa-snippet.sh %}
```

Well, we have WikiBooks database indexed locally.

### Search <a name="query"></a>
Let's do a test query:

```bash
# Do a match query that returns top-10 documents and its total count
summa-cli 0.0.0.0:8082 search '[{"index_alias": "books", "query": {"match": {"value": "astronomy"}}, "collectors": [{"top_docs": {"limit": 10}}, {"count": {}}]}]'
```

You will see response containing found documents.

## Further reading
- [Core components](/summa/core)
- [API References](/summa/apis)