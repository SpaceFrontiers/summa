# Create index schema in file
cat << EOF > es-wiki-schema.json
{% include es-wiki-schema.json %}
EOF

# (Optional) Delete previously created index
curl -XDELETE localhost:9200/books

# Create ES index
curl -H "Content-Type: application/json" -XPUT -d @es-wiki-schema.json localhost:9200/books

# Patch dump for ES 8
gsed -i 's/"_type":"books"/"_index":"books"/g' enwikibooks.json

# Import dump into ES
cat enwikibooks.json | parallel --pipe -L 2 -N 2000 -j3 'curl -H "Content-Type: application/json" -s http://localhost:9200/books/_bulk --data-binary @-'