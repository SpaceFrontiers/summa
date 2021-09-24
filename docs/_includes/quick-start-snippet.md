### Table of Contents
- [Setup](#setup)
- [Fill With Documents](#fill)
- [Query](#query)

### Setup <a name="setup"></a>
Both server `summa-server` and client `aiosumma` are distributed through the package systems, `Cargo` and `pip`.
Also, there is a prebuilt `summa-server` Docker image hosted on Dockerhub.

#### Aiosumma

```bash
# (Optional) Create virtual env for `aiosumma`
python3 -m venv venv
source venv/bin/acticate

# Install aiosumma
pip3 install -U aiosumma

# Install spaCy models
pip3 install -U https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0-py3-none-any.whl
```

#### Summa Server

```bash
# Create diectory for storing index
mkdir data

# Pull actual image for `summa-server`
docker pull izihawa/summa-server:testing

# Generate config for `summa-server`
docker run izihawa/summa-server:testing generate-config -d /data \
-g 0.0.0.0:8082 -m 0.0.0.0:8084 > summa.yaml

# Launch `summa-server`
docker run -v $(pwd)/summa.yaml:/summa.yaml -v $(pwd)/data:/data \
-p 8082:8082 -p 8084:8084 \
izihawa/summa-server:testing serve /summa.yaml
```

### Fill With Documents <a name="fill"></a>
```bash
{% include download-dump-snippet.sh %}

{% include import-data-to-summa-snippet.sh %}
```
### Query <a name="query"></a>
```bash
# Do a match query that returns top-10 documents and its total count
summa-cli localhost:8082 - search page '{"match": {"value": "astronomy"}}' '[{"top_docs": {"limit": 10}}, {"count": {}}]'
```
