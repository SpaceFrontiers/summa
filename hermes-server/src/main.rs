//! Hermes gRPC Search Server

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use parking_lot::RwLock;
use tonic::{Request, Response, Status, transport::Server};
use tracing::info;

use hermes_core::{
    BooleanQuery, BoostQuery, Document, FieldType as CoreFieldType, FieldValue as CoreFieldValue,
    FsDirectory, Index, IndexConfig, IndexWriter, Query, Schema, TermQuery, search_segment,
};

pub mod proto {
    tonic::include_proto!("hermes");
}

use proto::field_value::Value;
use proto::index_service_server::{IndexService, IndexServiceServer};
use proto::query::Query as ProtoQueryType;
use proto::search_service_server::{SearchService, SearchServiceServer};
use proto::*;

/// Hermes gRPC Search Server
#[derive(Parser, Debug)]
#[command(name = "hermes-server")]
#[command(about = "A high-performance async search server")]
struct Args {
    /// Address to bind to
    #[arg(short, long, default_value = "0.0.0.0:50051")]
    addr: String,

    /// Data directory for indexes
    #[arg(short, long, default_value = "./data")]
    data_dir: PathBuf,
}

/// Index registry holding all open indexes
struct IndexRegistry {
    indexes: RwLock<HashMap<String, Arc<Index<FsDirectory>>>>,
    writers: RwLock<HashMap<String, Arc<tokio::sync::Mutex<IndexWriter<FsDirectory>>>>>,
    data_dir: PathBuf,
}

impl IndexRegistry {
    fn new(data_dir: PathBuf) -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
            writers: RwLock::new(HashMap::new()),
            data_dir,
        }
    }

    async fn get_or_open_index(&self, name: &str) -> Result<Arc<Index<FsDirectory>>, Status> {
        // Check if already open
        if let Some(index) = self.indexes.read().get(name) {
            return Ok(Arc::clone(index));
        }

        // Try to open from disk
        let index_path = self.data_dir.join(name);
        if !index_path.exists() {
            return Err(Status::not_found(format!("Index '{}' not found", name)));
        }

        let dir = FsDirectory::new(&index_path);

        let config = IndexConfig::default();
        let index = Index::open(dir, config)
            .await
            .map_err(|e| Status::internal(format!("Failed to open index: {}", e)))?;

        let index = Arc::new(index);
        self.indexes
            .write()
            .insert(name.to_string(), Arc::clone(&index));
        Ok(index)
    }

    async fn create_index(&self, name: &str, schema: Schema) -> Result<(), Status> {
        let index_path = self.data_dir.join(name);

        if index_path.exists() {
            return Err(Status::already_exists(format!(
                "Index '{}' already exists",
                name
            )));
        }

        std::fs::create_dir_all(&index_path)
            .map_err(|e| Status::internal(format!("Failed to create directory: {}", e)))?;

        let dir = FsDirectory::new(&index_path);

        let config = IndexConfig::default();
        let writer = IndexWriter::create(dir.clone(), schema, config.clone())
            .await
            .map_err(|e| Status::internal(format!("Failed to create index: {}", e)))?;

        // Open the index for reading
        let index = Index::open(dir, config)
            .await
            .map_err(|e| Status::internal(format!("Failed to open created index: {}", e)))?;

        self.indexes
            .write()
            .insert(name.to_string(), Arc::new(index));
        self.writers
            .write()
            .insert(name.to_string(), Arc::new(tokio::sync::Mutex::new(writer)));

        Ok(())
    }

    async fn get_writer(
        &self,
        name: &str,
    ) -> Result<Arc<tokio::sync::Mutex<IndexWriter<FsDirectory>>>, Status> {
        if let Some(writer) = self.writers.read().get(name) {
            return Ok(Arc::clone(writer));
        }

        // Need to open writer
        let index_path = self.data_dir.join(name);
        if !index_path.exists() {
            return Err(Status::not_found(format!("Index '{}' not found", name)));
        }

        let dir = FsDirectory::new(&index_path);

        let config = IndexConfig::default();
        let writer = IndexWriter::open(dir, config)
            .await
            .map_err(|e| Status::internal(format!("Failed to open writer: {}", e)))?;

        let writer = Arc::new(tokio::sync::Mutex::new(writer));
        self.writers
            .write()
            .insert(name.to_string(), Arc::clone(&writer));
        Ok(writer)
    }
}

/// Search service implementation
struct SearchServiceImpl {
    registry: Arc<IndexRegistry>,
}

#[tonic::async_trait]
impl SearchService for SearchServiceImpl {
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        let index = self.registry.get_or_open_index(&req.index_name).await?;

        let query = req
            .query
            .ok_or_else(|| Status::invalid_argument("Query is required"))?;

        let core_query = convert_query(&query, index.schema())
            .map_err(|e| Status::invalid_argument(format!("Invalid query: {}", e)))?;

        let limit = if req.limit == 0 {
            10
        } else {
            req.limit as usize
        };

        // Search across all segments
        let mut all_results = Vec::new();
        for segment in index.segment_readers() {
            let results = search_segment(&segment, core_query.as_ref(), limit)
                .await
                .map_err(|e| Status::internal(format!("Search failed: {}", e)))?;

            for result in results {
                all_results.push((
                    result.doc_id + segment.doc_id_offset(),
                    result.score,
                    segment.clone(),
                ));
            }
        }

        // Sort by score descending
        all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(limit);

        // Convert to response
        let mut hits = Vec::new();
        for (doc_id, score, segment) in all_results {
            let local_doc_id = doc_id - segment.doc_id_offset();
            let mut fields = HashMap::new();

            if !req.fields_to_load.is_empty()
                && let Ok(Some(doc)) = segment.doc(local_doc_id).await
            {
                for field_name in &req.fields_to_load {
                    if let Some(field) = index.schema().get_field(field_name)
                        && let Some(value) = doc.get_first(field)
                    {
                        fields.insert(field_name.clone(), convert_field_value(value));
                    }
                }
            }

            hits.push(SearchHit {
                doc_id,
                score,
                fields,
            });
        }

        let took_ms = start.elapsed().as_millis() as u64;

        Ok(Response::new(SearchResponse {
            hits,
            total_hits: index.num_docs(),
            took_ms,
        }))
    }

    async fn get_document(
        &self,
        request: Request<GetDocumentRequest>,
    ) -> Result<Response<GetDocumentResponse>, Status> {
        let req = request.into_inner();
        let index = self.registry.get_or_open_index(&req.index_name).await?;

        let doc = index
            .doc(req.doc_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to get document: {}", e)))?
            .ok_or_else(|| Status::not_found("Document not found"))?;

        let mut fields = HashMap::new();
        for (field, value) in doc.field_values() {
            if let Some(entry) = index.schema().get_field_entry(*field) {
                fields.insert(entry.name.clone(), convert_field_value(value));
            }
        }

        Ok(Response::new(GetDocumentResponse { fields }))
    }

    async fn get_index_info(
        &self,
        request: Request<GetIndexInfoRequest>,
    ) -> Result<Response<GetIndexInfoResponse>, Status> {
        let req = request.into_inner();
        let index = self.registry.get_or_open_index(&req.index_name).await?;

        let schema = convert_schema_to_proto(index.schema());

        Ok(Response::new(GetIndexInfoResponse {
            index_name: req.index_name,
            num_docs: index.num_docs(),
            num_segments: index.segment_readers().len() as u32,
            schema: Some(schema),
        }))
    }
}

/// Index service implementation
struct IndexServiceImpl {
    registry: Arc<IndexRegistry>,
}

#[tonic::async_trait]
impl IndexService for IndexServiceImpl {
    async fn create_index(
        &self,
        request: Request<CreateIndexRequest>,
    ) -> Result<Response<CreateIndexResponse>, Status> {
        let req = request.into_inner();

        let proto_schema = req
            .schema
            .ok_or_else(|| Status::invalid_argument("Schema is required"))?;

        let schema = convert_proto_to_schema(&proto_schema)
            .map_err(|e| Status::invalid_argument(format!("Invalid schema: {}", e)))?;

        self.registry.create_index(&req.index_name, schema).await?;

        info!(index_name = %req.index_name, "Created index");

        Ok(Response::new(CreateIndexResponse { success: true }))
    }

    async fn index_documents(
        &self,
        request: Request<tonic::Streaming<IndexDocumentRequest>>,
    ) -> Result<Response<IndexDocumentsResponse>, Status> {
        let mut stream = request.into_inner();
        let mut indexed_count = 0u32;
        let mut current_index: Option<String> = None;
        let mut writer: Option<Arc<tokio::sync::Mutex<IndexWriter<FsDirectory>>>> = None;

        while let Some(req) = stream.message().await? {
            // Get or switch writer if index changed
            if current_index.as_ref() != Some(&req.index_name) {
                current_index = Some(req.index_name.clone());
                writer = Some(self.registry.get_writer(&req.index_name).await?);
            }

            let w = writer.as_ref().unwrap();
            let index = self.registry.get_or_open_index(&req.index_name).await?;

            let doc = convert_proto_to_document(&req.fields, index.schema())
                .map_err(|e| Status::invalid_argument(format!("Invalid document: {}", e)))?;

            w.lock()
                .await
                .add_document(doc)
                .await
                .map_err(|e| Status::internal(format!("Failed to index document: {}", e)))?;

            indexed_count += 1;
        }

        Ok(Response::new(IndexDocumentsResponse { indexed_count }))
    }

    async fn commit(
        &self,
        request: Request<CommitRequest>,
    ) -> Result<Response<CommitResponse>, Status> {
        let req = request.into_inner();
        let writer = self.registry.get_writer(&req.index_name).await?;

        writer
            .lock()
            .await
            .commit()
            .await
            .map_err(|e| Status::internal(format!("Commit failed: {}", e)))?;

        // Reload index to see new segments
        let index = self.registry.get_or_open_index(&req.index_name).await?;
        index
            .reload()
            .await
            .map_err(|e| Status::internal(format!("Reload failed: {}", e)))?;

        info!(index_name = %req.index_name, "Committed");

        Ok(Response::new(CommitResponse {
            success: true,
            num_docs: index.num_docs(),
        }))
    }

    async fn force_merge(
        &self,
        request: Request<ForceMergeRequest>,
    ) -> Result<Response<ForceMergeResponse>, Status> {
        let req = request.into_inner();
        let writer = self.registry.get_writer(&req.index_name).await?;

        writer
            .lock()
            .await
            .force_merge()
            .await
            .map_err(|e| Status::internal(format!("Merge failed: {}", e)))?;

        let index = self.registry.get_or_open_index(&req.index_name).await?;
        index
            .reload()
            .await
            .map_err(|e| Status::internal(format!("Reload failed: {}", e)))?;

        info!(index_name = %req.index_name, "Force merged");

        Ok(Response::new(ForceMergeResponse {
            success: true,
            num_segments: index.segment_readers().len() as u32,
        }))
    }

    async fn delete_index(
        &self,
        request: Request<DeleteIndexRequest>,
    ) -> Result<Response<DeleteIndexResponse>, Status> {
        let req = request.into_inner();

        // Remove from registry
        self.registry.indexes.write().remove(&req.index_name);
        self.registry.writers.write().remove(&req.index_name);

        // Delete directory
        let index_path = self.registry.data_dir.join(&req.index_name);
        if index_path.exists() {
            std::fs::remove_dir_all(&index_path)
                .map_err(|e| Status::internal(format!("Failed to delete index: {}", e)))?;
        }

        info!(index_name = %req.index_name, "Deleted index");

        Ok(Response::new(DeleteIndexResponse { success: true }))
    }
}

// Helper functions for conversion

fn convert_query(query: &proto::Query, schema: &Schema) -> Result<Box<dyn Query>, String> {
    match &query.query {
        Some(ProtoQueryType::Term(term_query)) => {
            let field = schema
                .get_field(&term_query.field)
                .ok_or_else(|| format!("Field '{}' not found", term_query.field))?;
            Ok(Box::new(TermQuery::text(field, &term_query.term)))
        }
        Some(ProtoQueryType::Boolean(bool_query)) => {
            let mut bq = BooleanQuery::new();
            for q in &bool_query.must {
                let inner = convert_query(q, schema)?;
                bq.must.push(inner);
            }
            for q in &bool_query.should {
                let inner = convert_query(q, schema)?;
                bq.should.push(inner);
            }
            for q in &bool_query.must_not {
                let inner = convert_query(q, schema)?;
                bq.must_not.push(inner);
            }
            Ok(Box::new(bq))
        }
        Some(ProtoQueryType::Boost(boost_query)) => {
            let inner = boost_query
                .query
                .as_ref()
                .ok_or_else(|| "Boost query requires inner query".to_string())?;
            let inner_query = convert_query(inner, schema)?;
            Ok(Box::new(BoostQuery {
                inner: inner_query,
                boost: boost_query.boost,
            }))
        }
        Some(ProtoQueryType::All(_)) => {
            // Match all - use a boolean query with no clauses that matches everything
            // For now, return an error as we don't have AllQuery implemented
            Err("AllQuery not yet implemented".to_string())
        }
        None => Err("Query type is required".to_string()),
    }
}

fn convert_field_value(value: &CoreFieldValue) -> proto::FieldValue {
    let v = match value {
        CoreFieldValue::Text(s) => Value::Text(s.clone()),
        CoreFieldValue::U64(n) => Value::U64(*n),
        CoreFieldValue::I64(n) => Value::I64(*n),
        CoreFieldValue::F64(n) => Value::F64(*n),
        CoreFieldValue::Bytes(b) => Value::BytesValue(b.clone()),
        CoreFieldValue::SparseVector { indices, values } => {
            Value::SparseVector(proto::SparseVector {
                indices: indices.clone(),
                values: values.clone(),
            })
        }
        CoreFieldValue::DenseVector(values) => Value::DenseVector(proto::DenseVector {
            values: values.clone(),
        }),
        CoreFieldValue::Json(json_val) => {
            Value::JsonValue(serde_json::to_string(json_val).unwrap_or_default())
        }
    };
    proto::FieldValue { value: Some(v) }
}

fn convert_schema_to_proto(schema: &Schema) -> proto::Schema {
    let fields = schema
        .fields()
        .map(|(_, entry)| proto::FieldDefinition {
            name: entry.name.clone(),
            field_type: match entry.field_type {
                CoreFieldType::Text => proto::FieldType::Text as i32,
                CoreFieldType::U64 => proto::FieldType::U64 as i32,
                CoreFieldType::I64 => proto::FieldType::I64 as i32,
                CoreFieldType::F64 => proto::FieldType::F64 as i32,
                CoreFieldType::Bytes => proto::FieldType::Bytes as i32,
                CoreFieldType::SparseVector => proto::FieldType::SparseVector as i32,
                CoreFieldType::DenseVector => proto::FieldType::DenseVector as i32,
                CoreFieldType::Json => proto::FieldType::Json as i32,
            },
            indexed: entry.indexed,
            stored: entry.stored,
        })
        .collect();

    proto::Schema { fields }
}

fn convert_proto_to_schema(proto_schema: &proto::Schema) -> Result<Schema, String> {
    let mut builder = hermes_core::schema::SchemaBuilder::default();

    for field in &proto_schema.fields {
        let field_type = proto::FieldType::try_from(field.field_type)
            .map_err(|_| format!("Invalid field type: {}", field.field_type))?;

        match field_type {
            proto::FieldType::Unspecified => return Err("Field type is required".to_string()),
            proto::FieldType::Text => {
                builder.add_text_field(&field.name, field.indexed, field.stored);
            }
            proto::FieldType::U64 => {
                builder.add_u64_field(&field.name, field.indexed, field.stored);
            }
            proto::FieldType::I64 => {
                builder.add_i64_field(&field.name, field.indexed, field.stored);
            }
            proto::FieldType::F64 => {
                builder.add_f64_field(&field.name, field.indexed, field.stored);
            }
            proto::FieldType::Bytes => {
                builder.add_bytes_field(&field.name, field.stored);
            }
            proto::FieldType::SparseVector => {
                builder.add_sparse_vector_field(&field.name, field.indexed, field.stored);
            }
            proto::FieldType::DenseVector => {
                // Default dimension of 0 - will be set from actual vectors
                builder.add_dense_vector_field(&field.name, 0, field.indexed, field.stored);
            }
            proto::FieldType::Json => {
                builder.add_json_field(&field.name, field.stored);
            }
        }
    }

    Ok(builder.build())
}

fn convert_proto_to_document(
    fields: &HashMap<String, proto::FieldValue>,
    schema: &Schema,
) -> Result<Document, String> {
    let mut doc = Document::new();

    for (name, value) in fields {
        let field = schema
            .get_field(name)
            .ok_or_else(|| format!("Field '{}' not found in schema", name))?;

        match &value.value {
            Some(Value::Text(s)) => doc.add_text(field, s),
            Some(Value::U64(n)) => doc.add_u64(field, *n),
            Some(Value::I64(n)) => doc.add_i64(field, *n),
            Some(Value::F64(n)) => doc.add_f64(field, *n),
            Some(Value::BytesValue(b)) => doc.add_bytes(field, b.clone()),
            Some(Value::SparseVector(sv)) => {
                doc.add_sparse_vector(field, sv.indices.clone(), sv.values.clone());
            }
            Some(Value::DenseVector(dv)) => {
                doc.add_dense_vector(field, dv.values.clone());
            }
            Some(Value::JsonValue(json_str)) => {
                let json_val: serde_json::Value = serde_json::from_str(json_str)
                    .map_err(|e| format!("Invalid JSON in field '{}': {}", name, e))?;
                doc.add_json(field, json_val);
            }
            None => return Err(format!("Field '{}' has no value", name)),
        }
    }

    Ok(doc)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("hermes_server=info".parse()?),
        )
        .init();

    let args = Args::parse();

    // Create data directory if needed
    std::fs::create_dir_all(&args.data_dir)?;

    let addr: SocketAddr = args.addr.parse()?;
    let registry = Arc::new(IndexRegistry::new(args.data_dir.clone()));

    let search_service = SearchServiceImpl {
        registry: Arc::clone(&registry),
    };

    let index_service = IndexServiceImpl {
        registry: Arc::clone(&registry),
    };

    info!("Starting Hermes server on {}", addr);
    info!("Data directory: {:?}", args.data_dir);

    Server::builder()
        .add_service(SearchServiceServer::new(search_service))
        .add_service(IndexServiceServer::new(index_service))
        .serve(addr)
        .await?;

    Ok(())
}
