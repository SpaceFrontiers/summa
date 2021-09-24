# Create index schema in file
cat << EOF > es-wiki-schema.json
{% include_relative files/es-wiki-schema.json %}
EOF

# (Optional) Delete previously created index
curl -XDELETE localhost:9200/page

# Create ES index
curl -H "Content-Type: application/json" -XPUT -d @es-wiki-schema.json localhost:9200/page

# Patch dump for ES 8
gsed -i 's/"_type":"page"/"_index":"page"/g' enwikibooks.json

# Import dump into ES
cat enwikibooks.json | parallel --pipe -L 2 -N 2000 -j3 'curl -H "Content-Type: application/json" -s http://localhost:9200/page/_bulk --data-binary @-'