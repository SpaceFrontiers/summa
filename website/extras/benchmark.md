---
title: Benchmark
parent: Extras
nav_order: 1
---

> Historical Summa 0.x guide. For Summa 2, use the [current documentation](https://github.com/SpaceFrontiers/summa/blob/main/docs/README.md).

Firstly, follow setup guide for [summa](https://spacefrontiers.github.io/summa/quick-start#setup) and [ES](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)

## Downloading Data

```bash
{% include download-dump-snippet.sh %}
```

## Preparing ES

```bash
{% include import-data-to-es-snippet.sh %}

# Do a test query
curl -H "Content-Type: application/json" -s http://localhost:9200/books/_search '{"query": { "match": {"message": {"query": "this is a test"}}}}'
```

## Preparing Summa

### Create Index

Summa is a schemaful search engines. It requires from you to define fields what you are going to use. Let's create
a schema for WikiBooks:

```bash
# Create index schema in file
cat << EOF > schema.yaml
{% include summa-wiki-schema.yaml %}
EOF
```

```bash
# Create index
summa-cli localhost:8082 - create-index-from-file schema.yaml
```

### Add documents

```bash
{% include import-data-to-summa-snippet.sh %}

# Do a match query that returns top-10 documents and its total count
summa-cli localhost:8082 search '{"index_alias": "books", "query": {"match": {"value": "astronomy"}}, "collectors": [{"top_docs": {"limit": 10}}, {"count": {}}]}'
```

## Benchmarking

ToDo
