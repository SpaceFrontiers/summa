# Hermes Schema Definition Language (SDL)

Hermes uses a simple, readable schema definition language for defining index schemas. This document describes the SDL syntax and features.

## Basic Syntax

An SDL file defines one or more indexes, each containing field definitions:

```
index <index_name> {
    field <field_name>: <field_type><tokenizer> [<attributes>]
    ...
}
```

## Example

```
# Article index schema
index articles {
    # Primary text field with English stemming
    field title: text<en_stem> [indexed, stored]

    # Body content with default tokenizer
    field body: text<default> [indexed]

    # Author name - no stemming needed
    field author: text [indexed, stored]

    # Publication timestamp
    field published_at: i64 [indexed, stored]

    # View count
    field views: u64 [indexed, stored]

    # Rating score
    field rating: f64 [indexed, stored]

    # Raw content hash (not indexed, just stored)
    field content_hash: bytes [stored]
}
```

## Field Types

| Type    | Aliases            | Description                                |
| ------- | ------------------ | ------------------------------------------ |
| `text`  | `string`, `str`    | UTF-8 text, tokenized for full-text search |
| `u64`   | `uint`, `unsigned` | Unsigned 64-bit integer                    |
| `i64`   | `int`, `integer`   | Signed 64-bit integer                      |
| `f64`   | `float`, `double`  | 64-bit floating point number               |
| `bytes` | `binary`, `blob`   | Raw binary data                            |

## Attributes

Attributes control how fields are processed and stored:

| Attribute | Description                                |
| --------- | ------------------------------------------ |
| `indexed` | Field is indexed for searching             |
| `stored`  | Field value is stored and can be retrieved |

### Attribute Syntax

Attributes are specified in square brackets after the field type:

```
field name: text [indexed, stored]    # Both indexed and stored
field name: text [indexed]            # Indexed only (not stored)
field name: text [stored]             # Stored only (not indexed)
field name: text                      # Default: indexed and stored
```

### Default Behavior

If no attributes are specified, fields default to **both indexed and stored**.

## Tokenizers

Text fields can specify a tokenizer using angle brackets after the type:

```
field title: text<en_stem> [indexed, stored]    # English stemmer
field body: text<german> [indexed]              # German stemmer
field name: text<default> [indexed, stored]     # Default (lowercase)
field raw: text [indexed, stored]               # Default tokenizer
```

### Available Tokenizers

| Name      | Aliases      | Description                                            |
| --------- | ------------ | ------------------------------------------------------ |
| `default` | `lowercase`  | Lowercase tokenizer (splits on whitespace, lowercases) |
| `simple`  | `raw`        | Simple whitespace tokenizer (no lowercasing)           |
| `en_stem` | `english`    | English Snowball stemmer                               |
| `de_stem` | `german`     | German Snowball stemmer                                |
| `fr_stem` | `french`     | French Snowball stemmer                                |
| `es_stem` | `spanish`    | Spanish Snowball stemmer                               |
| `it_stem` | `italian`    | Italian Snowball stemmer                               |
| `pt_stem` | `portuguese` | Portuguese Snowball stemmer                            |
| `ru_stem` | `russian`    | Russian Snowball stemmer                               |
| `ar_stem` | `arabic`     | Arabic Snowball stemmer                                |
| `da_stem` | `danish`     | Danish Snowball stemmer                                |
| `nl_stem` | `dutch`      | Dutch Snowball stemmer                                 |
| `fi_stem` | `finnish`    | Finnish Snowball stemmer                               |
| `el_stem` | `greek`      | Greek Snowball stemmer                                 |
| `hu_stem` | `hungarian`  | Hungarian Snowball stemmer                             |
| `no_stem` | `norwegian`  | Norwegian Snowball stemmer                             |
| `ro_stem` | `romanian`   | Romanian Snowball stemmer                              |
| `sv_stem` | `swedish`    | Swedish Snowball stemmer                               |
| `ta_stem` | `tamil`      | Tamil Snowball stemmer                                 |
| `tr_stem` | `turkish`    | Turkish Snowball stemmer                               |

### Custom Tokenizers

You can register custom tokenizers programmatically:

```rust
use hermes_core::{TokenizerRegistry, LowercaseTokenizer};

let registry = TokenizerRegistry::new();
registry.register("my_tokenizer", MyCustomTokenizer::new());
```

## Comments

Line comments start with `#`:

```
# This is a comment
index articles {
    # Title field for searching
    field title: text [indexed, stored]
}
```

## Multiple Indexes

A single SDL file can define multiple indexes:

```
index articles {
    field title: text [indexed, stored]
    field body: text [indexed]
}

index users {
    field name: text [indexed, stored]
    field email: text [indexed, stored]
    field created_at: i64 [indexed, stored]
}
```

## CLI Usage

### Create index from SDL file

```bash
hermes-index create -i ./myindex -s schema.sdl
```

### Create index from inline SDL

```bash
hermes-index init -i ./myindex -s 'index test { field title: text [indexed, stored] }'
```

## Grammar (PEG)

The SDL is parsed using [pest](https://pest.rs/). The complete grammar:

```pest
file = { SOI ~ index_def+ ~ EOI }

index_def = { "index" ~ identifier ~ "{" ~ field_def* ~ "}" }

field_def = { "field" ~ identifier ~ ":" ~ field_type ~ tokenizer_spec? ~ attributes? }

field_type = {
    "text" | "string" | "str" |
    "u64" | "uint" | "unsigned" |
    "i64" | "int" | "integer" |
    "f64" | "float" | "double" |
    "bytes" | "binary" | "blob"
}

tokenizer_spec = { "<" ~ identifier ~ ">" }

attributes = { "[" ~ attribute ~ ("," ~ attribute)* ~ "]" }
attribute = { "indexed" | "stored" }

identifier = @{ (ASCII_ALPHA | "_") ~ (ASCII_ALPHANUMERIC | "_")* }

WHITESPACE = _{ " " | "\t" | "\r" | "\n" }
COMMENT = _{ "#" ~ (!"\n" ~ ANY)* }
```

## Best Practices

1. **Use descriptive field names** - Field names should clearly indicate their purpose
2. **Only store what you need** - Use `[indexed]` without `stored` for large fields you only search but don't retrieve
3. **Use appropriate types** - Use numeric types for numbers to enable range queries
4. **Comment your schemas** - Add comments to explain field purposes
5. **Group related indexes** - Keep related indexes in the same SDL file
