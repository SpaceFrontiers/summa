---
title: Schema
parent: Core
nav_order: 3
---
## Schema
Summa stores and index objects named `Document`s. Every `Document` may be represented as json object with named fields:
```json 
{
  "title": "Forward the Foundation", 
  "text": "\"Hari,\" said Yugo Amaryl, \"that your friend Demerzel is in deep trouble.\" He emphasized the word \"friend\" very lighdy and with unmistakable air of distaste.", 
  "timestamp": 725839200
}
```
Schema describes a list of fields of every `Document` that will be indexed and become available for searching in Summa.
Let's look at the example that corresponds to the document above.
```yaml 
- name: title
  type: text
  options:
    indexing:
      fieldnorms: true
      record: position
      tokenizer: default
    stored: true
- name: text
  type: text
  options:
    indexing:
      fieldnorms: true
      record: position
      tokenizer: default
    stored: true
- name: timestamp
  type: date
  options:
    fast: single
    fieldnorms: false
    indexed: true
    stored: true
```
Every field has name, type and options describing indexing options.

### Available Types
#### bytes
Data should be passed as base64 encoded
#### date
#### f64
#### facet
#### i64
#### json
#### text
#### u64

### Options
#### fast
Marks fast fields. Fast fields are stored in separate columnar storage allowing fast access to the value of the field by document ID.
#### fieldnorms
#### indexed
#### indexing
#### stored
Stored set to `true` means value of field should be stored (not just indexed for search) for further retrieval.