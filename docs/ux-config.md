# Summa Web UX Configuration DSL

The `ux.dsl` file lives alongside the Summa index (next to `schema.json`) and configures how the web interface renders search results.

## File Location

```
/ipfs/<CID>/
├── schema.json
├── segments.json
├── ux.dsl          <-- UX configuration
└── segment_0/
    └── ...
```

## DSL Format

The DSL uses a simple, readable format inspired by CSS and configuration languages.

### Example

```dsl
# Search engine metadata
title "Product Catalog Search"
short_name "Products"  # Short name for connection lists
description "Search our product database"
placeholder "Search products by name, SKU, or description..."

# Result layout configuration
layout {
  # Which fields to display and in what order
  columns [title, price, category, sku]

  # How to render each field
  field title {
    label "Product Name"
    style bold
    width 40%
  }

  field price {
    label "Price"
    format currency
    prefix "$"
    width 15%
  }

  field category {
    label "Category"
    style tag
    color blue
    width 20%
  }

  field sku {
    label "SKU"
    style mono
    width 25%
  }
}

# Click actions - make fields clickable
actions {
  # Click on title opens product page
  click title {
    url "/products/{id}"
  }

  # Click on category filters by category
  click category {
    url "/search?category={category}"
  }

  # Click on SKU copies to clipboard
  click sku {
    action copy
    value "{sku}"
  }

  # External link example
  click external_link {
    url "{url}"
    target _blank
  }
}

# Row-level click action (optional)
row_click {
  url "/products/{id}"
}

# Custom CSS classes (optional)
styles {
  result_card "shadow-sm hover:shadow-md transition-shadow"
  highlight "bg-yellow-100"
}
```

## DSL Reference

### Metadata

| Directive     | Description                                             | Example                           |
| ------------- | ------------------------------------------------------- | --------------------------------- |
| `title`       | Search engine title displayed in header                 | `title "My Search"`               |
| `short_name`  | Short name for connection lists (defaults to IPNS name) | `short_name "Products"`           |
| `description` | Optional description                                    | `description "Search our data"`   |
| `placeholder` | Search input placeholder text                           | `placeholder "Type to search..."` |
| `logo`        | URL to logo image                                       | `logo "/logo.png"`                |

### Layout Block

```dsl
layout {
  columns [field1, field2, field3]  # Fields to show, in order

  field <name> {
    label "Display Name"    # Label shown above value (omit for no label)
    width 25%               # Column width (%, px, or auto)
    style bold|italic|mono|tag|link  # Visual style
    format text|number|currency|date|bytes  # Data format
    prefix "$"              # Text prefix
    suffix " USD"           # Text suffix
    color red|blue|green|gray|...  # Text/tag color
    truncate 50             # Max characters before truncation
    default "N/A"           # Default if field is empty
  }
}
```

### Actions Block

```dsl
actions {
  click <field_name> {
    url "<url_template>"    # URL with {field} placeholders
    target _self|_blank     # Link target (default: _self)
    action navigate|copy|custom  # Action type
    value "<template>"      # Value for copy action
  }
}
```

### URL Templates

Use `{field_name}` to interpolate field values:

```dsl
url "/products/{id}"
url "https://example.com/item/{sku}?ref={category}"
url "{external_url}"  # Use field value directly as URL
```

### Row Click

```dsl
row_click {
  url "<url_template>"
  target _self|_blank
}
```

### Styles Block (Optional)

```dsl
styles {
  result_card "<tailwind classes>"
  highlight "<tailwind classes>"
  header "<tailwind classes>"
}
```

## Parsing Rules

1. Lines starting with `#` are comments
2. Strings are quoted with `"` or `'`
3. Arrays use `[item1, item2, ...]` syntax
4. Blocks use `{ ... }` syntax
5. Whitespace is ignored (indentation is for readability)
6. **Field names are case-sensitive** and must match the document schema exactly (e.g., `content` not `Content`)

## Default Behavior

If no `ux.dsl` file is present:

- Title defaults to "Summa Search"
- All fields from schema are displayed
- No click actions
- Default styling applied
