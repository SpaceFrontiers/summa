use serde_json::json;
use summa_proto::proto;
use summa_proto::proto::index_api_client::IndexApiClient;
use summa_proto::proto::search_api_client::SearchApiClient;

static SCHEMA: &str = r#"
- name: title
  type: text
  options:
    indexing:
      fieldnorms: true
      record: position
      tokenizer: summa
    stored: true
- name: body
  type: text
  options:
    indexing:
      fieldnorms: true
      record: position
      tokenizer: summa
    stored: true
"#;

#[tokio::main]
async fn main() -> Result<(), tonic::Status> {
    let summa_conn = tonic::transport::Endpoint::new("grpc://127.0.0.1:8082")
        .expect("incorrect endpoint")
        .connect()
        .await
        .expect("cannot connect");
    let mut index_api_client = IndexApiClient::new(summa_conn.clone());
    let mut search_api_client = SearchApiClient::new(summa_conn);

    index_api_client
        .create_index(proto::CreateIndexRequest {
            index_name: "test_index".to_string(),
            index_engine: Some(proto::create_index_request::IndexEngine::File(proto::CreateFileEngineRequest {})),
            schema: SCHEMA.to_string(),
            compression: proto::Compression::Zstd.into(),
            index_attributes: Some(proto::IndexAttributes {
                unique_fields: vec!["title".to_string()],
                ..Default::default()
            }),
            ..Default::default()
        })
        .await?;

    index_api_client
        .index_document(proto::IndexDocumentRequest {
            index_name: "test_index".to_string(),
            document: serde_json::to_vec(&json!({
                "title": "Game of Thrones",
                "body": r#"Game of Thrones is an American fantasy drama television series created
                by David Benioff and D. B. Weiss for HBO. It is an adaptation of A Song of Ice and Fire,
                a series of fantasy novels by George R. R. Martin, the first of which is A Game of Thrones.
                The show was shot in the United Kingdom, Canada, Croatia, Iceland, Malta, Morocco, and
                Spain. It premiered on HBO in the United States on April 17, 2011, and concluded on May 19, 2019,
                with 73 episodes broadcast over eight seasons.
                "#,
            }))
            .unwrap(),
        })
        .await?;
    index_api_client
        .index_document(proto::IndexDocumentRequest {
            index_name: "test_index".to_string(),
            document: serde_json::to_vec(&json!({
                "title": "Breaking Bad",
                "body": r#"Breaking Bad is an American crime drama television series created and produced
                by Vince Gilligan. Set and filmed in Albuquerque, New Mexico, the series follows Walter
                White (Bryan Cranston), an underpaid, overqualified, and dispirited high-school chemistry
                teacher who is struggling with a recent diagnosis of stage-three lung cancer. White turns
                to a life of crime and partners with a former student, Jesse Pinkman (Aaron Paul), to
                produce and distribute methamphetamine to secure his family's financial future before he dies,
                while navigating the dangers of the criminal underworld. The show aired on AMC from January 20, 2008,
                to September 29, 2013, consisting of five seasons for a total of 62 episodes."#,
            }))
            .unwrap(),
        })
        .await?;

    index_api_client
        .commit_index(proto::CommitIndexRequest {
            index_name: "test_index".to_string(),
        })
        .await?;
    let search_response = search_api_client
        .search(proto::SearchRequest {
            index_queries: vec![proto::IndexQuery {
                index_alias: "test_index".to_string(),
                query: Some(proto::Query {
                    query: Some(proto::query::Query::Match(proto::MatchQuery {
                        value: "game".to_string(),
                        default_fields: vec!["title".to_string(), "body".to_string()],
                    })),
                }),
                collectors: vec![
                    proto::Collector {
                        collector: Some(proto::collector::Collector::TopDocs(proto::TopDocsCollector {
                            limit: 10,
                            ..Default::default()
                        })),
                    },
                    proto::Collector {
                        collector: Some(proto::collector::Collector::Count(proto::CountCollector {})),
                    },
                ],
                is_fieldnorms_scoring_enabled: None,
            }],
            tags: Default::default(),
        })
        .await?;
    println!("{:?}", search_response);
    Ok(())
}
