# Download sample dataset
CURRENT_DUMP=$(curl -s -L "https://dumps.wikimedia.org/other/cirrussearch/current" | grep -oh '\"enwikibooks.*\content.json\.gz\"' | tr -d '"')
wget "https://dumps.wikimedia.org/other/cirrussearch/current/$CURRENT_DUMP" -O enwikibooks.json.gz
gunzip enwikibooks.json.gz