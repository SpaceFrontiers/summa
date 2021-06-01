curl -H "Content-Type: application/json" -XPUT -d @es-wiki-schema.json localhost:9200/page
gsed -i 's/"_type":"page"/"_index":"page"/g' enwikibooks-20220523-cirrussearch-content.json
cat -n 1000 enwikibooks-20220523-cirrussearch-content.json | parallel --pipe -L 2 -N 2000 -j3 'curl -H "Content-Type: application/json" -s http://localhost:9200/page/_bulk --data-binary @-'

curl -H "Content-Type: application/json" -s http://localhost:9200/page/_search '{"query": { "match": {"message": {"query": "this is a test"}}}}'