### Summa

Summa is a tiny embeddedable search engine.

Main features
1. GRPC/HTTP API
2. Fast textual search
3. Space-efficient storage for documents

### Schema format

```yaml
---
# yamllint disable rule:key-ordering
# Default fields for non-specific search queries
default_fields: ["body", "title"]
# Is schema enabled?
enabled: true
# Unique field used for upserting. 
# Putting document with the same key field will lead to overwriting document
key_field: "id"
# Fields that can have multiple values
multi_fields: ["tags"]
# Unique name of the schema
name: products
# List of schema fields. Keys are the same as in Tantivy
schema:
  - name: id
    type: i64
    options:
      fast: single
      indexed: true
      stored: true
  - name: body
    type: text
    options:
      indexing:
        record: position
        tokenizer: summa
      stored: true
  - name: tags
    type: text
    options:
      indexing:
        record: position
        tokenizer: summa
      stored: true
  - name: title
    type: text
    options:
      indexing:
        record: position
        tokenizer: summa
      stored: true
```

### HTTP API v1

#### `Search`
`GET /v1/{schema_name}/search/?query=lord`

#### `Asyncronous add`
`PUT /v1/{schema_name}/` -H 'Content-Type: application/json' --data '{"id": 1, "title": "Lord of the Rings"}'

#### `Commit`
`POST /v1/{schema_name}/commit/`