# Create index schema in file
cat << EOF > schema.yaml
{% include_relative files/summa-wiki-schema.yaml %}
EOF

# Create index
summa-cli localhost:8082 - create-index-from-file schema.yaml

# Upload documents
awk 'NR%2==0' en-wiki-books.json | summa-cli localhost:8082 - index-document-stream page

# Commit index to make them searchable
summa-cli localhost:8082 - commit-index page --commit-mode Sync