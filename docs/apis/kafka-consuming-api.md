---
title: Kafka Consuming
parent: APIs
---
## Consumer API
Summa can ingest documents through Kafka.
The core concept is Consumer that can be created through API. 
Consumer operated in a separate thread and transfers messages from Kafka topic into the selected index.
At the moment it is the most performant way to index large number of documents.