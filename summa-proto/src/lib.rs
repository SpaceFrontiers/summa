pub mod proto_traits;

#[allow(clippy::derive_partial_eq_without_eq)]
/// Protobuf messages for communicating with Summa
///
/// ```rust,no_run
/// use summa_proto::proto;
/// use summa_proto::proto::search_api_client::SearchApiClient;
///
/// #[tokio::main]
/// async fn main() {
///     let summa_conn = tonic::transport::Endpoint::new("grpc://127.0.0.1:8082")
///         .expect("incorrect endpoint")
///         .connect()
///         .await
///         .expect("cannot connect");
///     let mut search_api_client = SearchApiClient::new(summa_conn);
///
///     let search_response = search_api_client
///         .search(proto::SearchRequest {
///             index_queries: vec![proto::IndexQuery {
///                 index_alias: "test_index".to_string(),
///                 query: Some(proto::Query {
///                     query: Some(proto::query::Query::Match(proto::MatchQuery {
///                         value: "game of thrones".to_string(),
///                         query_parser_config: Some(proto::QueryParserConfig {
///                             default_fields: vec!["title".to_string()],
///                             ..Default::default()
///                         }),
///                         ..Default::default()
///                     })),
///                 }),
///                 collectors: vec![
///                     proto::Collector {
///                         collector: Some(proto::collector::Collector::TopDocs(proto::TopDocsCollector {
///                             limit: 10,
///                             ..Default::default()
///                         })),
///                     },
///                     proto::Collector {
///                         collector: Some(proto::collector::Collector::Count(proto::CountCollector {})),
///                     }
///                 ],
///                 is_fieldnorms_scoring_enabled: None,
///             }],
///             tags: Default::default(),
///         })
///         .await
///         .expect("cannot search");
/// }
/// ```
pub mod proto {
    #[cfg(feature = "grpc")]
    pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("summa");
    pub mod dag_pb {
        include!(concat!(env!("OUT_DIR"), "/dag_pb.rs"));
    }
    pub mod unixfs {
        include!(concat!(env!("OUT_DIR"), "/unixfs.rs"));
    }
    #[cfg(feature = "grpc")]
    tonic::include_proto!("summa.proto");
    #[cfg(not(feature = "grpc"))]
    include!(concat!(env!("OUT_DIR"), "/summa.proto.rs"));
}
