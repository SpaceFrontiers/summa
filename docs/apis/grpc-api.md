---
title: GRPC
parent: APIs
---
## GRPC API

- [consumer_service.proto](#consumer_service-proto)
    - [ConsumerApi](#summa-proto-ConsumerApi)
  
    - [Consumer](#summa-proto-Consumer)
    - [CreateConsumerRequest](#summa-proto-CreateConsumerRequest)
    - [CreateConsumerResponse](#summa-proto-CreateConsumerResponse)
    - [DeleteConsumerRequest](#summa-proto-DeleteConsumerRequest)
    - [DeleteConsumerResponse](#summa-proto-DeleteConsumerResponse)
    - [GetConsumerRequest](#summa-proto-GetConsumerRequest)
    - [GetConsumerResponse](#summa-proto-GetConsumerResponse)
    - [GetConsumersRequest](#summa-proto-GetConsumersRequest)
    - [GetConsumersResponse](#summa-proto-GetConsumersResponse)
  
- [dag_pb.proto](#dag_pb-proto)
    - [PBLink](#dag_pb-PBLink)
    - [PBNode](#dag_pb-PBNode)
  
- [index_service.proto](#index_service-proto)
    - [IndexApi](#summa-proto-IndexApi)
  
    - [AttachFileEngineRequest](#summa-proto-AttachFileEngineRequest)
    - [AttachIndexRequest](#summa-proto-AttachIndexRequest)
    - [AttachIndexResponse](#summa-proto-AttachIndexResponse)
    - [AttachRemoteEngineRequest](#summa-proto-AttachRemoteEngineRequest)
    - [CacheConfig](#summa-proto-CacheConfig)
    - [CommitIndexRequest](#summa-proto-CommitIndexRequest)
    - [CommitIndexResponse](#summa-proto-CommitIndexResponse)
    - [CopyDocumentsRequest](#summa-proto-CopyDocumentsRequest)
    - [CopyDocumentsResponse](#summa-proto-CopyDocumentsResponse)
    - [CopyIndexRequest](#summa-proto-CopyIndexRequest)
    - [CopyIndexResponse](#summa-proto-CopyIndexResponse)
    - [CreateFileEngineRequest](#summa-proto-CreateFileEngineRequest)
    - [CreateIndexRequest](#summa-proto-CreateIndexRequest)
    - [CreateIndexResponse](#summa-proto-CreateIndexResponse)
    - [CreateMemoryEngineRequest](#summa-proto-CreateMemoryEngineRequest)
    - [DeleteDocumentsRequest](#summa-proto-DeleteDocumentsRequest)
    - [DeleteDocumentsResponse](#summa-proto-DeleteDocumentsResponse)
    - [DeleteIndexRequest](#summa-proto-DeleteIndexRequest)
    - [DeleteIndexResponse](#summa-proto-DeleteIndexResponse)
    - [DocumentsRequest](#summa-proto-DocumentsRequest)
    - [DocumentsResponse](#summa-proto-DocumentsResponse)
    - [FileEngineConfig](#summa-proto-FileEngineConfig)
    - [GetIndexRequest](#summa-proto-GetIndexRequest)
    - [GetIndexResponse](#summa-proto-GetIndexResponse)
    - [GetIndicesAliasesRequest](#summa-proto-GetIndicesAliasesRequest)
    - [GetIndicesAliasesResponse](#summa-proto-GetIndicesAliasesResponse)
    - [GetIndicesAliasesResponse.IndicesAliasesEntry](#summa-proto-GetIndicesAliasesResponse-IndicesAliasesEntry)
    - [GetIndicesRequest](#summa-proto-GetIndicesRequest)
    - [GetIndicesResponse](#summa-proto-GetIndicesResponse)
    - [IndexAttributes](#summa-proto-IndexAttributes)
    - [IndexDescription](#summa-proto-IndexDescription)
    - [IndexDocumentOperation](#summa-proto-IndexDocumentOperation)
    - [IndexDocumentRequest](#summa-proto-IndexDocumentRequest)
    - [IndexDocumentResponse](#summa-proto-IndexDocumentResponse)
    - [IndexDocumentStreamRequest](#summa-proto-IndexDocumentStreamRequest)
    - [IndexDocumentStreamResponse](#summa-proto-IndexDocumentStreamResponse)
    - [IndexEngineConfig](#summa-proto-IndexEngineConfig)
    - [IndexOperation](#summa-proto-IndexOperation)
    - [LogMergePolicy](#summa-proto-LogMergePolicy)
    - [MappedField](#summa-proto-MappedField)
    - [MemoryEngineConfig](#summa-proto-MemoryEngineConfig)
    - [MergePolicy](#summa-proto-MergePolicy)
    - [MergeSegmentsRequest](#summa-proto-MergeSegmentsRequest)
    - [MergeSegmentsResponse](#summa-proto-MergeSegmentsResponse)
    - [PrimaryKey](#summa-proto-PrimaryKey)
    - [RemoteEngineConfig](#summa-proto-RemoteEngineConfig)
    - [RemoteEngineConfig.HeadersTemplateEntry](#summa-proto-RemoteEngineConfig-HeadersTemplateEntry)
    - [SetIndexAliasRequest](#summa-proto-SetIndexAliasRequest)
    - [SetIndexAliasResponse](#summa-proto-SetIndexAliasResponse)
    - [SortByField](#summa-proto-SortByField)
    - [TemporalMergePolicy](#summa-proto-TemporalMergePolicy)
    - [VacuumIndexRequest](#summa-proto-VacuumIndexRequest)
    - [VacuumIndexResponse](#summa-proto-VacuumIndexResponse)
    - [WarmupIndexRequest](#summa-proto-WarmupIndexRequest)
    - [WarmupIndexResponse](#summa-proto-WarmupIndexResponse)
  
    - [Compression](#summa-proto-Compression)
    - [ConflictStrategy](#summa-proto-ConflictStrategy)
  
- [query.proto](#query-proto)
    - [AggregationCollector](#summa-proto-AggregationCollector)
    - [AggregationCollectorOutput](#summa-proto-AggregationCollectorOutput)
    - [AllQuery](#summa-proto-AllQuery)
    - [BooleanQuery](#summa-proto-BooleanQuery)
    - [BooleanSubquery](#summa-proto-BooleanSubquery)
    - [BoostQuery](#summa-proto-BoostQuery)
    - [Collector](#summa-proto-Collector)
    - [CollectorOutput](#summa-proto-CollectorOutput)
    - [CountCollector](#summa-proto-CountCollector)
    - [CountCollectorOutput](#summa-proto-CountCollectorOutput)
    - [CustomOrder](#summa-proto-CustomOrder)
    - [DisjunctionMaxQuery](#summa-proto-DisjunctionMaxQuery)
    - [DocumentsCollectorOutput](#summa-proto-DocumentsCollectorOutput)
    - [EmptyQuery](#summa-proto-EmptyQuery)
    - [ExactMatchesPromoter](#summa-proto-ExactMatchesPromoter)
    - [ExistsQuery](#summa-proto-ExistsQuery)
    - [FacetCollector](#summa-proto-FacetCollector)
    - [FacetCollectorOutput](#summa-proto-FacetCollectorOutput)
    - [FacetCollectorOutput.FacetCountsEntry](#summa-proto-FacetCollectorOutput-FacetCountsEntry)
    - [Highlight](#summa-proto-Highlight)
    - [Key](#summa-proto-Key)
    - [MatchQuery](#summa-proto-MatchQuery)
    - [MatchQueryBooleanShouldMode](#summa-proto-MatchQueryBooleanShouldMode)
    - [MatchQueryDisjuctionMaxMode](#summa-proto-MatchQueryDisjuctionMaxMode)
    - [MoreLikeThisQuery](#summa-proto-MoreLikeThisQuery)
    - [MorphologyConfig](#summa-proto-MorphologyConfig)
    - [NerMatchesPromoter](#summa-proto-NerMatchesPromoter)
    - [PhraseQuery](#summa-proto-PhraseQuery)
    - [Query](#summa-proto-Query)
    - [QueryParserConfig](#summa-proto-QueryParserConfig)
    - [QueryParserConfig.FieldAliasesEntry](#summa-proto-QueryParserConfig-FieldAliasesEntry)
    - [QueryParserConfig.FieldBoostsEntry](#summa-proto-QueryParserConfig-FieldBoostsEntry)
    - [QueryParserConfig.MorphologyConfigsEntry](#summa-proto-QueryParserConfig-MorphologyConfigsEntry)
    - [QueryParserConfig.TermFieldMapperConfigsEntry](#summa-proto-QueryParserConfig-TermFieldMapperConfigsEntry)
    - [RandomDocument](#summa-proto-RandomDocument)
    - [Range](#summa-proto-Range)
    - [RangeQuery](#summa-proto-RangeQuery)
    - [RegexQuery](#summa-proto-RegexQuery)
    - [ReservoirSamplingCollector](#summa-proto-ReservoirSamplingCollector)
    - [ReservoirSamplingCollectorOutput](#summa-proto-ReservoirSamplingCollectorOutput)
    - [Score](#summa-proto-Score)
    - [ScoredDocument](#summa-proto-ScoredDocument)
    - [ScoredDocument.SnippetsEntry](#summa-proto-ScoredDocument-SnippetsEntry)
    - [Scorer](#summa-proto-Scorer)
    - [SearchResponse](#summa-proto-SearchResponse)
    - [Snippet](#summa-proto-Snippet)
    - [TermFieldMapperConfig](#summa-proto-TermFieldMapperConfig)
    - [TermQuery](#summa-proto-TermQuery)
    - [TopDocsCollector](#summa-proto-TopDocsCollector)
    - [TopDocsCollector.SnippetConfigsEntry](#summa-proto-TopDocsCollector-SnippetConfigsEntry)
  
    - [Occur](#summa-proto-Occur)
  
- [reflection_service.proto](#reflection_service-proto)
    - [ReflectionApi](#summa-proto-ReflectionApi)
  
    - [GetTopTermsRequest](#summa-proto-GetTopTermsRequest)
    - [GetTopTermsResponse](#summa-proto-GetTopTermsResponse)
    - [GetTopTermsResponse.PerSegmentEntry](#summa-proto-GetTopTermsResponse-PerSegmentEntry)
    - [SegmentTerms](#summa-proto-SegmentTerms)
    - [TermInfo](#summa-proto-TermInfo)
  
- [search_service.proto](#search_service-proto)
    - [SearchApi](#summa-proto-SearchApi)
  
    - [SearchRequest](#summa-proto-SearchRequest)
  
- [unixfs.proto](#unixfs-proto)
    - [Data](#unixfs-Data)
    - [Metadata](#unixfs-Metadata)
  
    - [Data.DataType](#unixfs-Data-DataType)
  
- [utils.proto](#utils-proto)
    - [Empty](#summa-proto-Empty)
  
    - [Order](#summa-proto-Order)
  



<a name="consumer_service-proto"></a>

## consumer_service.proto



<a name="summa-proto-Consumer"></a>

### Consumer
Consumer description


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| consumer_name | [string](#string) |  | Consumer name |
| index_name | [string](#string) |  | Summa `index_name` |






<a name="summa-proto-CreateConsumerRequest"></a>

### CreateConsumerRequest
Request describe how new Consumer should be created


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| bootstrap_servers | [string](#string) | repeated | Kafka boostrap servers |
| group_id | [string](#string) |  | Kafka group ID |
| index_name | [string](#string) |  | Summa `index_name` which will ingest data from Kafka topics |
| consumer_name | [string](#string) |  | Consumer name, used for further referencing consumer in API and configs |
| topics | [string](#string) | repeated | List of topics to consume |






<a name="summa-proto-CreateConsumerResponse"></a>

### CreateConsumerResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| consumer | [Consumer](#summa-proto-Consumer) |  |  |






<a name="summa-proto-DeleteConsumerRequest"></a>

### DeleteConsumerRequest



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| consumer_name | [string](#string) |  |  |






<a name="summa-proto-DeleteConsumerResponse"></a>

### DeleteConsumerResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| consumer_name | [string](#string) |  |  |






<a name="summa-proto-GetConsumerRequest"></a>

### GetConsumerRequest



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  |  |
| consumer_name | [string](#string) |  |  |






<a name="summa-proto-GetConsumerResponse"></a>

### GetConsumerResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| consumer | [Consumer](#summa-proto-Consumer) |  |  |






<a name="summa-proto-GetConsumersRequest"></a>

### GetConsumersRequest







<a name="summa-proto-GetConsumersResponse"></a>

### GetConsumersResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| consumers | [Consumer](#summa-proto-Consumer) | repeated |  |





 <!-- end messages -->

 <!-- end enums -->

 <!-- end HasExtensions -->


<a name="summa-proto-ConsumerApi"></a>

### ConsumerApi
Manage ingestion data from Kafka

| Method Name | Request Type | Response Type | Description |
| ----------- | ------------ | ------------- | ------------|
| create_consumer | [CreateConsumerRequest](#summa-proto-CreateConsumerRequest) | [CreateConsumerResponse](#summa-proto-CreateConsumerResponse) | Create a new consumer |
| get_consumer | [GetConsumerRequest](#summa-proto-GetConsumerRequest) | [GetConsumerResponse](#summa-proto-GetConsumerResponse) | Get a single consumer |
| get_consumers | [GetConsumersRequest](#summa-proto-GetConsumersRequest) | [GetConsumersResponse](#summa-proto-GetConsumersResponse) | Get a list of all consumers |
| delete_consumer | [DeleteConsumerRequest](#summa-proto-DeleteConsumerRequest) | [DeleteConsumerResponse](#summa-proto-DeleteConsumerResponse) | Remove a consumer |

 <!-- end services -->



<a name="dag_pb-proto"></a>

## dag_pb.proto



<a name="dag_pb-PBLink"></a>

### PBLink



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| hash | [bytes](#bytes) | optional | binary CID (with no multibase prefix) of the target object |
| name | [string](#string) | optional | UTF-8 string name |
| t_size | [uint64](#uint64) | optional | cumulative size of target object |






<a name="dag_pb-PBNode"></a>

### PBNode



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| links | [PBLink](#dag_pb-PBLink) | repeated | refs to other objects |
| data | [bytes](#bytes) | optional | opaque user data |





 <!-- end messages -->

 <!-- end enums -->

 <!-- end HasExtensions -->

 <!-- end services -->



<a name="index_service-proto"></a>

## index_service.proto



<a name="summa-proto-AttachFileEngineRequest"></a>

### AttachFileEngineRequest
Attach file engine request






<a name="summa-proto-AttachIndexRequest"></a>

### AttachIndexRequest
Attach index request


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  | Index name for attaching |
| file | [AttachFileEngineRequest](#summa-proto-AttachFileEngineRequest) |  |  |
| remote | [AttachRemoteEngineRequest](#summa-proto-AttachRemoteEngineRequest) |  |  |
| merge_policy | [MergePolicy](#summa-proto-MergePolicy) |  |  |
| query_parser_config | [QueryParserConfig](#summa-proto-QueryParserConfig) |  |  |






<a name="summa-proto-AttachIndexResponse"></a>

### AttachIndexResponse
Description of the attached index


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index | [IndexDescription](#summa-proto-IndexDescription) |  |  |






<a name="summa-proto-AttachRemoteEngineRequest"></a>

### AttachRemoteEngineRequest
Attach remote engine request


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| config | [RemoteEngineConfig](#summa-proto-RemoteEngineConfig) |  |  |






<a name="summa-proto-CacheConfig"></a>

### CacheConfig



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| cache_size | [uint64](#uint64) |  | Total cache size in bytes |






<a name="summa-proto-CommitIndexRequest"></a>

### CommitIndexRequest
Store the state of index to the storage


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  |  |






<a name="summa-proto-CommitIndexResponse"></a>

### CommitIndexResponse
Returned data from the commit command


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| elapsed_secs | [double](#double) |  | Pure time spent for committing |






<a name="summa-proto-CopyDocumentsRequest"></a>

### CopyDocumentsRequest
Copy documents from one index to another. Their schemes must be compatible


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| source_index_name | [string](#string) |  | Where documents should be taken from |
| target_index_name | [string](#string) |  | Where documents should be copied to |
| conflict_strategy | [ConflictStrategy](#summa-proto-ConflictStrategy) | optional | How to deal with conflicts on unique fields. Recommended to set to DoNothing for large updates and maintain uniqueness in your application |






<a name="summa-proto-CopyDocumentsResponse"></a>

### CopyDocumentsResponse
Copy documents response


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| elapsed_secs | [double](#double) |  |  |
| copied_documents | [uint32](#uint32) |  |  |






<a name="summa-proto-CopyIndexRequest"></a>

### CopyIndexRequest
Request that changes index engine. Currently possible to convert File to IPFS


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| source_index_name | [string](#string) |  | Name of index that will be migrated. It will be left intact after migration. |
| target_index_name | [string](#string) |  | Name of index that will be created |
| file | [CreateFileEngineRequest](#summa-proto-CreateFileEngineRequest) |  |  |
| memory | [CreateMemoryEngineRequest](#summa-proto-CreateMemoryEngineRequest) |  |  |
| merge_policy | [MergePolicy](#summa-proto-MergePolicy) |  |  |






<a name="summa-proto-CopyIndexResponse"></a>

### CopyIndexResponse
Response describing migrated index


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index | [IndexDescription](#summa-proto-IndexDescription) |  |  |






<a name="summa-proto-CreateFileEngineRequest"></a>

### CreateFileEngineRequest







<a name="summa-proto-CreateIndexRequest"></a>

### CreateIndexRequest
Request for index creation


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  | Index name |
| file | [CreateFileEngineRequest](#summa-proto-CreateFileEngineRequest) |  |  |
| memory | [CreateMemoryEngineRequest](#summa-proto-CreateMemoryEngineRequest) |  |  |
| schema | [string](#string) |  | Index schema in Tantivy format |
| compression | [Compression](#summa-proto-Compression) |  | Compression for store |
| blocksize | [uint32](#uint32) | optional | Size of store blocks |
| sort_by_field | [SortByField](#summa-proto-SortByField) | optional | Field for sorting |
| index_attributes | [IndexAttributes](#summa-proto-IndexAttributes) |  | Optional index fields |
| merge_policy | [MergePolicy](#summa-proto-MergePolicy) |  | Merge policy |
| query_parser_config | [QueryParserConfig](#summa-proto-QueryParserConfig) |  |  |






<a name="summa-proto-CreateIndexResponse"></a>

### CreateIndexResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index | [IndexDescription](#summa-proto-IndexDescription) |  |  |






<a name="summa-proto-CreateMemoryEngineRequest"></a>

### CreateMemoryEngineRequest







<a name="summa-proto-DeleteDocumentsRequest"></a>

### DeleteDocumentsRequest



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  |  |
| query | [Query](#summa-proto-Query) |  |  |






<a name="summa-proto-DeleteDocumentsResponse"></a>

### DeleteDocumentsResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| deleted_documents | [uint64](#uint64) |  |  |






<a name="summa-proto-DeleteIndexRequest"></a>

### DeleteIndexRequest



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  |  |






<a name="summa-proto-DeleteIndexResponse"></a>

### DeleteIndexResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| deleted_index_name | [string](#string) |  |  |






<a name="summa-proto-DocumentsRequest"></a>

### DocumentsRequest
Request a stream of all documents from the index


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  |  |
| fields | [string](#string) | repeated |  |






<a name="summa-proto-DocumentsResponse"></a>

### DocumentsResponse
Single document from the index


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| document | [string](#string) |  |  |






<a name="summa-proto-FileEngineConfig"></a>

### FileEngineConfig



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| path | [string](#string) |  |  |






<a name="summa-proto-GetIndexRequest"></a>

### GetIndexRequest



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  |  |






<a name="summa-proto-GetIndexResponse"></a>

### GetIndexResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index | [IndexDescription](#summa-proto-IndexDescription) |  |  |






<a name="summa-proto-GetIndicesAliasesRequest"></a>

### GetIndicesAliasesRequest







<a name="summa-proto-GetIndicesAliasesResponse"></a>

### GetIndicesAliasesResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| indices_aliases | [GetIndicesAliasesResponse.IndicesAliasesEntry](#summa-proto-GetIndicesAliasesResponse-IndicesAliasesEntry) | repeated |  |






<a name="summa-proto-GetIndicesAliasesResponse-IndicesAliasesEntry"></a>

### GetIndicesAliasesResponse.IndicesAliasesEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [string](#string) |  |  |






<a name="summa-proto-GetIndicesRequest"></a>

### GetIndicesRequest







<a name="summa-proto-GetIndicesResponse"></a>

### GetIndicesResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_names | [string](#string) | repeated |  |






<a name="summa-proto-IndexAttributes"></a>

### IndexAttributes



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| created_at | [uint64](#uint64) |  | Timestamp when index has been created |
| unique_fields | [string](#string) | repeated | Unique fields of the index. Summa maintains unique constraint on them and uses for deduplicating data |
| multi_fields | [string](#string) | repeated | Multi fields is ones that may have multiple values and processed as lists. All other fields will be forcefully converted to singular value |
| description | [string](#string) | optional | Text index description |
| conflict_strategy | [ConflictStrategy](#summa-proto-ConflictStrategy) |  |  |
| mapped_fields | [MappedField](#summa-proto-MappedField) | repeated |  |






<a name="summa-proto-IndexDescription"></a>

### IndexDescription
Description containing `Index` metadata fields


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  |  |
| index_aliases | [string](#string) | repeated | All index aliases |
| index_engine | [IndexEngineConfig](#summa-proto-IndexEngineConfig) |  |  |
| num_docs | [uint64](#uint64) |  | The number of committed documents |
| compression | [Compression](#summa-proto-Compression) |  | Used compression for `store` |
| index_attributes | [IndexAttributes](#summa-proto-IndexAttributes) |  | All custom index attributes |






<a name="summa-proto-IndexDocumentOperation"></a>

### IndexDocumentOperation
Indexing operations that contains document serialized in JSON format


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| document | [bytes](#bytes) |  |  |






<a name="summa-proto-IndexDocumentRequest"></a>

### IndexDocumentRequest



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  |  |
| document | [bytes](#bytes) |  |  |






<a name="summa-proto-IndexDocumentResponse"></a>

### IndexDocumentResponse







<a name="summa-proto-IndexDocumentStreamRequest"></a>

### IndexDocumentStreamRequest



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  |  |
| documents | [bytes](#bytes) | repeated |  |
| conflict_strategy | [ConflictStrategy](#summa-proto-ConflictStrategy) | optional |  |






<a name="summa-proto-IndexDocumentStreamResponse"></a>

### IndexDocumentStreamResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| elapsed_secs | [double](#double) |  |  |
| success_docs | [uint64](#uint64) |  |  |
| failed_docs | [uint64](#uint64) |  |  |






<a name="summa-proto-IndexEngineConfig"></a>

### IndexEngineConfig
Description of the `IndexEngine` responsible for managing files in the persistent storage


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| file | [FileEngineConfig](#summa-proto-FileEngineConfig) |  |  |
| memory | [MemoryEngineConfig](#summa-proto-MemoryEngineConfig) |  |  |
| remote | [RemoteEngineConfig](#summa-proto-RemoteEngineConfig) |  |  |
| merge_policy | [MergePolicy](#summa-proto-MergePolicy) |  | Merge policy |
| query_parser_config | [QueryParserConfig](#summa-proto-QueryParserConfig) |  |  |






<a name="summa-proto-IndexOperation"></a>

### IndexOperation
Message that should be put in Kafka for ingesting by Summa consumers


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_document | [IndexDocumentOperation](#summa-proto-IndexDocumentOperation) |  |  |






<a name="summa-proto-LogMergePolicy"></a>

### LogMergePolicy
Merge policy for implementing [LogMergePolicy](https://docs.rs/tantivy/latest/tantivy/merge_policy/struct.LogMergePolicy.html)


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| is_frozen | [bool](#bool) |  | Set if once merged segment should be left intact |






<a name="summa-proto-MappedField"></a>

### MappedField



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| source_field | [string](#string) |  |  |
| target_field | [string](#string) |  |  |






<a name="summa-proto-MemoryEngineConfig"></a>

### MemoryEngineConfig



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| schema | [string](#string) |  | Schema of the index for memory engine |






<a name="summa-proto-MergePolicy"></a>

### MergePolicy
Merge policy that describes how to merge committed segments


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| log | [LogMergePolicy](#summa-proto-LogMergePolicy) |  |  |
| temporal | [TemporalMergePolicy](#summa-proto-TemporalMergePolicy) |  |  |






<a name="summa-proto-MergeSegmentsRequest"></a>

### MergeSegmentsRequest



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  |  |
| segment_ids | [string](#string) | repeated |  |






<a name="summa-proto-MergeSegmentsResponse"></a>

### MergeSegmentsResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| segment_id | [string](#string) | optional |  |






<a name="summa-proto-PrimaryKey"></a>

### PrimaryKey
Possible primary keys


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| str | [string](#string) |  |  |
| i64 | [int64](#int64) |  |  |






<a name="summa-proto-RemoteEngineConfig"></a>

### RemoteEngineConfig
Remote HTTP engine config


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| method | [string](#string) |  | Which method should be used to request remote endpoint |
| url_template | [string](#string) |  | URL template which will be used to generate real URL by variables substitution |
| headers_template | [RemoteEngineConfig.HeadersTemplateEntry](#summa-proto-RemoteEngineConfig-HeadersTemplateEntry) | repeated | Headers template which will be used to generate real URL by variables substitution |
| cache_config | [CacheConfig](#summa-proto-CacheConfig) |  | Description of the cache for the engine |
| timeout_ms | [uint32](#uint32) | optional | Timeout for the request |






<a name="summa-proto-RemoteEngineConfig-HeadersTemplateEntry"></a>

### RemoteEngineConfig.HeadersTemplateEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [string](#string) |  |  |






<a name="summa-proto-SetIndexAliasRequest"></a>

### SetIndexAliasRequest



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_alias | [string](#string) |  |  |
| index_name | [string](#string) |  |  |






<a name="summa-proto-SetIndexAliasResponse"></a>

### SetIndexAliasResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| old_index_name | [string](#string) | optional | If set, equals to the previous alias of the index |






<a name="summa-proto-SortByField"></a>

### SortByField



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| field | [string](#string) |  |  |
| order | [Order](#summa-proto-Order) |  |  |






<a name="summa-proto-TemporalMergePolicy"></a>

### TemporalMergePolicy
Merge policy for compressing old segments


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| merge_older_then_secs | [uint64](#uint64) |  |  |






<a name="summa-proto-VacuumIndexRequest"></a>

### VacuumIndexRequest



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  |  |
| excluded_segments | [string](#string) | repeated |  |






<a name="summa-proto-VacuumIndexResponse"></a>

### VacuumIndexResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| freed_space_bytes | [uint64](#uint64) |  |  |






<a name="summa-proto-WarmupIndexRequest"></a>

### WarmupIndexRequest



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  |  |
| is_full | [bool](#bool) |  | If set to false, only term dictionaries will be warmed, otherwise the entire index will be read. |






<a name="summa-proto-WarmupIndexResponse"></a>

### WarmupIndexResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| elapsed_secs | [double](#double) |  | Time spent in warming operation |





 <!-- end messages -->


<a name="summa-proto-Compression"></a>

### Compression
Compression library for store, implies on both performance and occupied disk space

| Name | Number | Description |
| ---- | ------ | ----------- |
| None | 0 |  |
| Brotli | 1 |  |
| Lz4 | 2 |  |
| Snappy | 3 |  |
| Zstd | 4 |  |
| Zstd7 | 5 |  |
| Zstd9 | 6 |  |
| Zstd14 | 7 |  |
| Zstd19 | 8 |  |
| Zstd22 | 9 |  |



<a name="summa-proto-ConflictStrategy"></a>

### ConflictStrategy


| Name | Number | Description |
| ---- | ------ | ----------- |
| DO_NOTHING | 0 |  |
| OVERWRITE_ALWAYS | 1 |  |
| OVERWRITE | 2 |  |
| MERGE | 3 |  |


 <!-- end enums -->

 <!-- end HasExtensions -->


<a name="summa-proto-IndexApi"></a>

### IndexApi
Manages indices

| Method Name | Request Type | Response Type | Description |
| ----------- | ------------ | ------------- | ------------|
| attach_index | [AttachIndexRequest](#summa-proto-AttachIndexRequest) | [AttachIndexResponse](#summa-proto-AttachIndexResponse) | Attaches index to Summa server. Attaching allows to incorporate and start using of downloaded or network indices |
| commit_index | [CommitIndexRequest](#summa-proto-CommitIndexRequest) | [CommitIndexResponse](#summa-proto-CommitIndexResponse) | Committing all collected writes to the index |
| copy_documents | [CopyDocumentsRequest](#summa-proto-CopyDocumentsRequest) | [CopyDocumentsResponse](#summa-proto-CopyDocumentsResponse) | Copy documents from one index to another |
| create_index | [CreateIndexRequest](#summa-proto-CreateIndexRequest) | [CreateIndexResponse](#summa-proto-CreateIndexResponse) | Creates new index from scratch |
| copy_index | [CopyIndexRequest](#summa-proto-CopyIndexRequest) | [CopyIndexResponse](#summa-proto-CopyIndexResponse) | Creates new index from scratch |
| delete_documents | [DeleteDocumentsRequest](#summa-proto-DeleteDocumentsRequest) | [DeleteDocumentsResponse](#summa-proto-DeleteDocumentsResponse) | Deletes single document from the index by its primary key (therefore, index must have primary key) |
| delete_index | [DeleteIndexRequest](#summa-proto-DeleteIndexRequest) | [DeleteIndexResponse](#summa-proto-DeleteIndexResponse) | Deletes index and physically removes file in the case of `FileEngine` |
| documents | [DocumentsRequest](#summa-proto-DocumentsRequest) | [DocumentsResponse](#summa-proto-DocumentsResponse) stream | Stream of all documents from the index |
| get_indices_aliases | [GetIndicesAliasesRequest](#summa-proto-GetIndicesAliasesRequest) | [GetIndicesAliasesResponse](#summa-proto-GetIndicesAliasesResponse) | Gets all existing index aliases |
| get_index | [GetIndexRequest](#summa-proto-GetIndexRequest) | [GetIndexResponse](#summa-proto-GetIndexResponse) | Gets index description |
| get_indices | [GetIndicesRequest](#summa-proto-GetIndicesRequest) | [GetIndicesResponse](#summa-proto-GetIndicesResponse) | Gets all existing index descriptions |
| index_document_stream | [IndexDocumentStreamRequest](#summa-proto-IndexDocumentStreamRequest) stream | [IndexDocumentStreamResponse](#summa-proto-IndexDocumentStreamResponse) | Adds document to the index in a streaming way |
| index_document | [IndexDocumentRequest](#summa-proto-IndexDocumentRequest) | [IndexDocumentResponse](#summa-proto-IndexDocumentResponse) | Adds document to the index |
| merge_segments | [MergeSegmentsRequest](#summa-proto-MergeSegmentsRequest) | [MergeSegmentsResponse](#summa-proto-MergeSegmentsResponse) | Merges multiple segments into a single one. Used for service purposes |
| set_index_alias | [SetIndexAliasRequest](#summa-proto-SetIndexAliasRequest) | [SetIndexAliasResponse](#summa-proto-SetIndexAliasResponse) | Sets or replaces existing index alias |
| vacuum_index | [VacuumIndexRequest](#summa-proto-VacuumIndexRequest) | [VacuumIndexResponse](#summa-proto-VacuumIndexResponse) | Removes deletions from all segments |
| warmup_index | [WarmupIndexRequest](#summa-proto-WarmupIndexRequest) | [WarmupIndexResponse](#summa-proto-WarmupIndexResponse) | Loads all hot parts of the index into the memory |

 <!-- end services -->



<a name="query-proto"></a>

## query.proto



<a name="summa-proto-AggregationCollector"></a>

### AggregationCollector



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| aggregations | [string](#string) |  |  |






<a name="summa-proto-AggregationCollectorOutput"></a>

### AggregationCollectorOutput



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| aggregation_results | [string](#string) |  |  |






<a name="summa-proto-AllQuery"></a>

### AllQuery







<a name="summa-proto-BooleanQuery"></a>

### BooleanQuery



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| subqueries | [BooleanSubquery](#summa-proto-BooleanSubquery) | repeated |  |






<a name="summa-proto-BooleanSubquery"></a>

### BooleanSubquery



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| occur | [Occur](#summa-proto-Occur) |  |  |
| query | [Query](#summa-proto-Query) |  |  |






<a name="summa-proto-BoostQuery"></a>

### BoostQuery



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| query | [Query](#summa-proto-Query) |  |  |
| score | [string](#string) |  |  |






<a name="summa-proto-Collector"></a>

### Collector
Collectors and CollectorOutputs


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| top_docs | [TopDocsCollector](#summa-proto-TopDocsCollector) |  |  |
| reservoir_sampling | [ReservoirSamplingCollector](#summa-proto-ReservoirSamplingCollector) |  |  |
| count | [CountCollector](#summa-proto-CountCollector) |  |  |
| facet | [FacetCollector](#summa-proto-FacetCollector) |  |  |
| aggregation | [AggregationCollector](#summa-proto-AggregationCollector) |  |  |






<a name="summa-proto-CollectorOutput"></a>

### CollectorOutput



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| documents | [DocumentsCollectorOutput](#summa-proto-DocumentsCollectorOutput) |  |  |
| count | [CountCollectorOutput](#summa-proto-CountCollectorOutput) |  |  |
| facet | [FacetCollectorOutput](#summa-proto-FacetCollectorOutput) |  |  |
| aggregation | [AggregationCollectorOutput](#summa-proto-AggregationCollectorOutput) |  |  |






<a name="summa-proto-CountCollector"></a>

### CountCollector







<a name="summa-proto-CountCollectorOutput"></a>

### CountCollectorOutput



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| count | [uint32](#uint32) |  |  |






<a name="summa-proto-CustomOrder"></a>

### CustomOrder



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [Empty](#summa-proto-Empty) |  |  |
| count | [Empty](#summa-proto-Empty) |  |  |
| sub_aggregation | [string](#string) |  |  |
| order | [Order](#summa-proto-Order) |  |  |






<a name="summa-proto-DisjunctionMaxQuery"></a>

### DisjunctionMaxQuery



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| disjuncts | [Query](#summa-proto-Query) | repeated |  |
| tie_breaker | [string](#string) |  |  |






<a name="summa-proto-DocumentsCollectorOutput"></a>

### DocumentsCollectorOutput



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| scored_documents | [ScoredDocument](#summa-proto-ScoredDocument) | repeated |  |
| has_next | [bool](#bool) |  |  |






<a name="summa-proto-EmptyQuery"></a>

### EmptyQuery







<a name="summa-proto-ExactMatchesPromoter"></a>

### ExactMatchesPromoter



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| slop | [uint32](#uint32) |  |  |
| boost | [float](#float) | optional |  |
| fields | [string](#string) | repeated |  |






<a name="summa-proto-ExistsQuery"></a>

### ExistsQuery



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| field | [string](#string) |  |  |






<a name="summa-proto-FacetCollector"></a>

### FacetCollector



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| field | [string](#string) |  |  |
| facets | [string](#string) | repeated |  |






<a name="summa-proto-FacetCollectorOutput"></a>

### FacetCollectorOutput



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| facet_counts | [FacetCollectorOutput.FacetCountsEntry](#summa-proto-FacetCollectorOutput-FacetCountsEntry) | repeated |  |






<a name="summa-proto-FacetCollectorOutput-FacetCountsEntry"></a>

### FacetCollectorOutput.FacetCountsEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [uint64](#uint64) |  |  |






<a name="summa-proto-Highlight"></a>

### Highlight



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| from | [uint32](#uint32) |  |  |
| to | [uint32](#uint32) |  |  |






<a name="summa-proto-Key"></a>

### Key



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| str | [string](#string) |  |  |
| f64 | [double](#double) |  |  |






<a name="summa-proto-MatchQuery"></a>

### MatchQuery



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| value | [string](#string) |  |  |
| query_parser_config | [QueryParserConfig](#summa-proto-QueryParserConfig) | optional |  |






<a name="summa-proto-MatchQueryBooleanShouldMode"></a>

### MatchQueryBooleanShouldMode







<a name="summa-proto-MatchQueryDisjuctionMaxMode"></a>

### MatchQueryDisjuctionMaxMode



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| tie_breaker | [float](#float) |  |  |






<a name="summa-proto-MoreLikeThisQuery"></a>

### MoreLikeThisQuery



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| document | [string](#string) |  |  |
| min_doc_frequency | [uint64](#uint64) | optional |  |
| max_doc_frequency | [uint64](#uint64) | optional |  |
| min_term_frequency | [uint64](#uint64) | optional |  |
| max_query_terms | [uint64](#uint64) | optional |  |
| min_word_length | [uint64](#uint64) | optional |  |
| max_word_length | [uint64](#uint64) | optional |  |
| boost | [string](#string) | optional |  |
| stop_words | [string](#string) | repeated |  |






<a name="summa-proto-MorphologyConfig"></a>

### MorphologyConfig



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| derive_tenses_coefficient | [float](#float) | optional |  |






<a name="summa-proto-NerMatchesPromoter"></a>

### NerMatchesPromoter



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| boost | [float](#float) | optional |  |
| fields | [string](#string) | repeated |  |






<a name="summa-proto-PhraseQuery"></a>

### PhraseQuery



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| field | [string](#string) |  |  |
| value | [string](#string) |  |  |
| slop | [uint32](#uint32) |  |  |






<a name="summa-proto-Query"></a>

### Query
Recursive query DSL


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| boolean | [BooleanQuery](#summa-proto-BooleanQuery) |  |  |
| match | [MatchQuery](#summa-proto-MatchQuery) |  |  |
| regex | [RegexQuery](#summa-proto-RegexQuery) |  |  |
| term | [TermQuery](#summa-proto-TermQuery) |  |  |
| phrase | [PhraseQuery](#summa-proto-PhraseQuery) |  |  |
| range | [RangeQuery](#summa-proto-RangeQuery) |  |  |
| all | [AllQuery](#summa-proto-AllQuery) |  |  |
| more_like_this | [MoreLikeThisQuery](#summa-proto-MoreLikeThisQuery) |  |  |
| boost | [BoostQuery](#summa-proto-BoostQuery) |  |  |
| disjunction_max | [DisjunctionMaxQuery](#summa-proto-DisjunctionMaxQuery) |  |  |
| empty | [EmptyQuery](#summa-proto-EmptyQuery) |  |  |
| exists | [ExistsQuery](#summa-proto-ExistsQuery) |  |  |






<a name="summa-proto-QueryParserConfig"></a>

### QueryParserConfig



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| field_aliases | [QueryParserConfig.FieldAliasesEntry](#summa-proto-QueryParserConfig-FieldAliasesEntry) | repeated |  |
| field_boosts | [QueryParserConfig.FieldBoostsEntry](#summa-proto-QueryParserConfig-FieldBoostsEntry) | repeated |  |
| term_field_mapper_configs | [QueryParserConfig.TermFieldMapperConfigsEntry](#summa-proto-QueryParserConfig-TermFieldMapperConfigsEntry) | repeated |  |
| term_limit | [uint32](#uint32) |  |  |
| default_fields | [string](#string) | repeated |  |
| boolean_should_mode | [MatchQueryBooleanShouldMode](#summa-proto-MatchQueryBooleanShouldMode) |  |  |
| disjuction_max_mode | [MatchQueryDisjuctionMaxMode](#summa-proto-MatchQueryDisjuctionMaxMode) |  |  |
| exact_matches_promoter | [ExactMatchesPromoter](#summa-proto-ExactMatchesPromoter) |  |  |
| removed_fields | [string](#string) | repeated |  |
| morphology_configs | [QueryParserConfig.MorphologyConfigsEntry](#summa-proto-QueryParserConfig-MorphologyConfigsEntry) | repeated |  |
| query_language | [string](#string) | optional |  |






<a name="summa-proto-QueryParserConfig-FieldAliasesEntry"></a>

### QueryParserConfig.FieldAliasesEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [string](#string) |  |  |






<a name="summa-proto-QueryParserConfig-FieldBoostsEntry"></a>

### QueryParserConfig.FieldBoostsEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [float](#float) |  |  |






<a name="summa-proto-QueryParserConfig-MorphologyConfigsEntry"></a>

### QueryParserConfig.MorphologyConfigsEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [MorphologyConfig](#summa-proto-MorphologyConfig) |  |  |






<a name="summa-proto-QueryParserConfig-TermFieldMapperConfigsEntry"></a>

### QueryParserConfig.TermFieldMapperConfigsEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [TermFieldMapperConfig](#summa-proto-TermFieldMapperConfig) |  |  |






<a name="summa-proto-RandomDocument"></a>

### RandomDocument



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| document | [string](#string) |  |  |
| score | [Score](#summa-proto-Score) |  |  |
| index_alias | [string](#string) |  |  |






<a name="summa-proto-Range"></a>

### Range



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| left | [string](#string) |  |  |
| right | [string](#string) |  |  |
| including_left | [bool](#bool) |  |  |
| including_right | [bool](#bool) |  |  |






<a name="summa-proto-RangeQuery"></a>

### RangeQuery



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| field | [string](#string) |  |  |
| value | [Range](#summa-proto-Range) |  |  |






<a name="summa-proto-RegexQuery"></a>

### RegexQuery



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| field | [string](#string) |  |  |
| value | [string](#string) |  |  |






<a name="summa-proto-ReservoirSamplingCollector"></a>

### ReservoirSamplingCollector



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| limit | [uint32](#uint32) |  |  |
| fields | [string](#string) | repeated |  |






<a name="summa-proto-ReservoirSamplingCollectorOutput"></a>

### ReservoirSamplingCollectorOutput



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| documents | [RandomDocument](#summa-proto-RandomDocument) | repeated |  |






<a name="summa-proto-Score"></a>

### Score



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| f64_score | [double](#double) |  |  |
| u64_score | [uint64](#uint64) |  |  |






<a name="summa-proto-ScoredDocument"></a>

### ScoredDocument



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| document | [string](#string) |  |  |
| score | [Score](#summa-proto-Score) |  |  |
| position | [uint32](#uint32) |  |  |
| snippets | [ScoredDocument.SnippetsEntry](#summa-proto-ScoredDocument-SnippetsEntry) | repeated |  |
| index_alias | [string](#string) |  |  |






<a name="summa-proto-ScoredDocument-SnippetsEntry"></a>

### ScoredDocument.SnippetsEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [Snippet](#summa-proto-Snippet) |  |  |






<a name="summa-proto-Scorer"></a>

### Scorer



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| eval_expr | [string](#string) |  |  |
| order_by | [string](#string) |  |  |






<a name="summa-proto-SearchResponse"></a>

### SearchResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| elapsed_secs | [double](#double) |  | Time spent inside of `search` handler |
| collector_outputs | [CollectorOutput](#summa-proto-CollectorOutput) | repeated | An array of collector outputs |






<a name="summa-proto-Snippet"></a>

### Snippet



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| fragment | [bytes](#bytes) |  |  |
| highlights | [Highlight](#summa-proto-Highlight) | repeated |  |
| html | [string](#string) |  |  |






<a name="summa-proto-TermFieldMapperConfig"></a>

### TermFieldMapperConfig



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| fields | [string](#string) | repeated |  |






<a name="summa-proto-TermQuery"></a>

### TermQuery



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| field | [string](#string) |  |  |
| value | [string](#string) |  |  |






<a name="summa-proto-TopDocsCollector"></a>

### TopDocsCollector



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| limit | [uint32](#uint32) |  |  |
| offset | [uint32](#uint32) |  |  |
| scorer | [Scorer](#summa-proto-Scorer) | optional |  |
| snippet_configs | [TopDocsCollector.SnippetConfigsEntry](#summa-proto-TopDocsCollector-SnippetConfigsEntry) | repeated |  |
| explain | [bool](#bool) |  |  |
| fields | [string](#string) | repeated |  |






<a name="summa-proto-TopDocsCollector-SnippetConfigsEntry"></a>

### TopDocsCollector.SnippetConfigsEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [uint32](#uint32) |  |  |





 <!-- end messages -->


<a name="summa-proto-Occur"></a>

### Occur


| Name | Number | Description |
| ---- | ------ | ----------- |
| should | 0 |  |
| must | 1 |  |
| must_not | 2 |  |


 <!-- end enums -->

 <!-- end HasExtensions -->

 <!-- end services -->



<a name="reflection_service-proto"></a>

## reflection_service.proto



<a name="summa-proto-GetTopTermsRequest"></a>

### GetTopTermsRequest



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_name | [string](#string) |  |  |
| field_name | [string](#string) |  |  |
| top_k | [uint32](#uint32) |  |  |






<a name="summa-proto-GetTopTermsResponse"></a>

### GetTopTermsResponse



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| per_segment | [GetTopTermsResponse.PerSegmentEntry](#summa-proto-GetTopTermsResponse-PerSegmentEntry) | repeated |  |






<a name="summa-proto-GetTopTermsResponse-PerSegmentEntry"></a>

### GetTopTermsResponse.PerSegmentEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [SegmentTerms](#summa-proto-SegmentTerms) |  |  |






<a name="summa-proto-SegmentTerms"></a>

### SegmentTerms



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| term_infos | [TermInfo](#summa-proto-TermInfo) | repeated |  |






<a name="summa-proto-TermInfo"></a>

### TermInfo



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [bytes](#bytes) |  |  |
| doc_freq | [uint32](#uint32) |  |  |





 <!-- end messages -->

 <!-- end enums -->

 <!-- end HasExtensions -->


<a name="summa-proto-ReflectionApi"></a>

### ReflectionApi
Analyzes indices

| Method Name | Request Type | Response Type | Description |
| ----------- | ------------ | ------------- | ------------|
| get_top_terms | [GetTopTermsRequest](#summa-proto-GetTopTermsRequest) | [GetTopTermsResponse](#summa-proto-GetTopTermsResponse) |  |

 <!-- end services -->



<a name="search_service-proto"></a>

## search_service.proto



<a name="summa-proto-SearchRequest"></a>

### SearchRequest



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| index_alias | [string](#string) |  | The index name or alias |
| query | [Query](#summa-proto-Query) |  | Query DSL. Use `MatchQuery` to pass a free-form query |
| collectors | [Collector](#summa-proto-Collector) | repeated | Every collector is responsible of processing and storing documents and/or their derivatives (like counters) to return them to the caller |
| is_fieldnorms_scoring_enabled | [bool](#bool) | optional | Is requiring fieldnorms needed for the query? |





 <!-- end messages -->

 <!-- end enums -->

 <!-- end HasExtensions -->


<a name="summa-proto-SearchApi"></a>

### SearchApi
Searches documents in the stored indices

| Method Name | Request Type | Response Type | Description |
| ----------- | ------------ | ------------- | ------------|
| search | [SearchRequest](#summa-proto-SearchRequest) | [SearchResponse](#summa-proto-SearchResponse) | Make search in Summa |

 <!-- end services -->



<a name="unixfs-proto"></a>

## unixfs.proto



<a name="unixfs-Data"></a>

### Data



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| type | [Data.DataType](#unixfs-Data-DataType) |  |  |
| data | [bytes](#bytes) | optional |  |
| filesize | [uint64](#uint64) | optional |  |
| blocksizes | [uint64](#uint64) | repeated |  |
| hashType | [uint64](#uint64) | optional |  |
| fanout | [uint64](#uint64) | optional |  |






<a name="unixfs-Metadata"></a>

### Metadata



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| MimeType | [string](#string) | optional |  |





 <!-- end messages -->


<a name="unixfs-Data-DataType"></a>

### Data.DataType


| Name | Number | Description |
| ---- | ------ | ----------- |
| Raw | 0 |  |
| Directory | 1 |  |
| File | 2 |  |
| Metadata | 3 |  |
| Symlink | 4 |  |
| HAMTShard | 5 |  |


 <!-- end enums -->

 <!-- end HasExtensions -->

 <!-- end services -->



<a name="utils-proto"></a>

## utils.proto



<a name="summa-proto-Empty"></a>

### Empty






 <!-- end messages -->


<a name="summa-proto-Order"></a>

### Order


| Name | Number | Description |
| ---- | ------ | ----------- |
| Asc | 0 |  |
| Desc | 1 |  |


 <!-- end enums -->

 <!-- end HasExtensions -->

 <!-- end services -->


