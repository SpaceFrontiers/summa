# summa

Fast full-text search server with following features:

- Simple GRPC API
- Indexing documents through Kafka

### Quickstart
#### Create index schema
```yaml
---
# yamllint disable rule:key-ordering
- name: id
  type: i64
  options:
    fast: single
    fieldnorms: false
    indexed: true
    stored: true
- name: body
  type: text
  options:
    indexing:
      fieldnorms: true
      record: position
      tokenizer: summa
    stored: true
- name: tags
  type: text
  options:
    indexing:
      fieldnorms: true
      record: position
      tokenizer: summa
    stored: true
- name: title
  type: text
  options:
    indexing:
      fieldnorms: true
      record: position
      tokenizer: summa
    stored: true
```
