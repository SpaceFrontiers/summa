# Upload a half of documents to Summa. You can upload remaining half by setting `awk 'NR%4==2'`
# It will take a while depending on the performance of your computer
awk 'NR%4==0' enwikibooks.json | summa-cli localhost:8082 - index-document-stream books

# Commit index to make them searchable
summa-cli localhost:8082 - commit-index books --commit-mode Sync