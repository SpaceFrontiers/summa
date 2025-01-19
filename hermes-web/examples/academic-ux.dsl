# Academic Search Engine UX Configuration
# Example configuration for scholarly content
# Note: Field names must match the document schema exactly (case-sensitive)

title "Academic Search"
short_name "Academic"
description "Search academic papers, books, and articles"
placeholder "Search papers, authors, topics..."

layout {
  columns [content, languages, issued_at, uris, id]

  content {
    format split_newline
    style bold
  }

  languages {
    label "Lang"
    format language_emoji
  }

  issued_at {
    label "Published"
    format unix_year
  }

  uris {
    label "Links"
    format uri_links
    style link
  }

  id {
    label "Download"
    format ipfs_files
    ipfs_formats [pdf, epub, djvu]
  }

  type {
    label "Type"
    style tag
    color blue
  }
}
