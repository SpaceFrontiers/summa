### Summa

Summa is a tiny embeddedable search engine.

Main features
1. GRPC/HTTP API
2. Fast textual search
3. Space-efficient storage for documents

### HTTP API v1

#### `Search`
`GET /v1/{schema_name}/search/?query=foobar`

#### `Add`
`PUT /v1/{schema_name}/`

#### `Commit`
`POST /v1/{schema_name}/commit/`