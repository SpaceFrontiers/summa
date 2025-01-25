//! Hermes gRPC Search Server

mod converters;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use parking_lot::RwLock;
use tonic::{Request, Response, Status, codec::CompressionEncoding, transport::Server};
use tracing::info;

use hermes_core::{
    FsDirectory, Index, IndexConfig, IndexWriter, Schema, parse_schema, search_segment,
};

pub mod proto {
    tonic::include_proto!("hermes");
}

use converters::{convert_field_value, convert_proto_to_document, convert_query, schema_to_sdl};
use proto::index_service_server::{IndexService, IndexServiceServer};
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

    /// Cache directory for HuggingFace models/tokenizers
    #[arg(short, long)]
    cache_dir: Option<PathBuf>,

    /// Max indexing memory (MB) before auto-flush (global across all builders)
    #[arg(long, default_value = "3072")]
    max_indexing_memory_mb: usize,

    /// Number of parallel indexing threads (defaults to CPU count)
    #[arg(long)]
    indexing_threads: Option<usize>,
}

/// Index registry holding all open indexes
struct IndexRegistry {
    indexes: RwLock<HashMap<String, Arc<Index<FsDirectory>>>>,
    writers: RwLock<HashMap<String, Arc<IndexWriter<FsDirectory>>>>,
    data_dir: PathBuf,
    config: IndexConfig,
}

impl IndexRegistry {
    fn new(data_dir: PathBuf, config: IndexConfig) -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
            writers: RwLock::new(HashMap::new()),
            data_dir,
            config,
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

        let index = Index::open(dir, self.config.clone())
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

        let writer = IndexWriter::create(dir.clone(), schema, self.config.clone())
            .await
            .map_err(|e| Status::internal(format!("Failed to create index: {}", e)))?;

        // Open the index for reading
        let index = Index::open(dir, self.config.clone())
            .await
            .map_err(|e| Status::internal(format!("Failed to open created index: {}", e)))?;

        self.indexes
            .write()
            .insert(name.to_string(), Arc::new(index));
        self.writers
            .write()
            .insert(name.to_string(), Arc::new(writer));

        Ok(())
    }

    async fn get_writer(&self, name: &str) -> Result<Arc<IndexWriter<FsDirectory>>, Status> {
        if let Some(writer) = self.writers.read().get(name) {
            return Ok(Arc::clone(writer));
        }

        // Need to open writer
        let index_path = self.data_dir.join(name);
        if !index_path.exists() {
            return Err(Status::not_found(format!("Index '{}' not found", name)));
        }

        let dir = FsDirectory::new(&index_path);

        let writer = IndexWriter::open(dir, self.config.clone())
            .await
            .map_err(|e| Status::internal(format!("Failed to open writer: {}", e)))?;

        let writer = Arc::new(writer);
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

        // Convert schema to SDL string
        let schema_str = schema_to_sdl(index.schema());

        Ok(Response::new(GetIndexInfoResponse {
            index_name: req.index_name,
            num_docs: index.num_docs(),
            num_segments: index.segment_readers().len() as u32,
            schema: schema_str,
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

        if req.schema.is_empty() {
            return Err(Status::invalid_argument("Schema is required"));
        }

        let schema = parse_schema(&req.schema)
            .map_err(|e| Status::invalid_argument(format!("Invalid schema: {}", e)))?;

        self.registry.create_index(&req.index_name, schema).await?;

        info!(index_name = %req.index_name, "Created index");

        Ok(Response::new(CreateIndexResponse { success: true }))
    }

    async fn batch_index_documents(
        &self,
        request: Request<BatchIndexDocumentsRequest>,
    ) -> Result<Response<BatchIndexDocumentsResponse>, Status> {
        let req = request.into_inner();

        let writer = self.registry.get_writer(&req.index_name).await?;
        let index = self.registry.get_or_open_index(&req.index_name).await?;
        let schema = index.schema();

        // Convert all documents first (this is CPU-bound, could be parallelized further)
        let mut documents = Vec::with_capacity(req.documents.len());
        let mut conversion_errors = 0u32;

        for named_doc in req.documents {
            match convert_proto_to_document(&named_doc.fields, schema) {
                Ok(doc) => documents.push(doc),
                Err(_) => conversion_errors += 1,
            }
        }

        // Use batch method for parallel document indexing
        // IndexWriter is internally thread-safe, no external lock needed
        let doc_count = documents.len() as u32;
        let indexed_count = writer.add_documents(documents).await.unwrap_or(0) as u32;
        let index_errors = doc_count - indexed_count;

        let error_count = conversion_errors + index_errors;

        info!(
            index_name = %req.index_name,
            indexed = indexed_count,
            errors = error_count,
            "Batch indexed documents"
        );

        Ok(Response::new(BatchIndexDocumentsResponse {
            indexed_count,
            error_count,
        }))
    }

    async fn index_documents(
        &self,
        request: Request<tonic::Streaming<IndexDocumentRequest>>,
    ) -> Result<Response<IndexDocumentsResponse>, Status> {
        let mut stream = request.into_inner();
        let mut indexed_count = 0u32;
        let mut current_index: Option<String> = None;
        let mut writer: Option<Arc<IndexWriter<FsDirectory>>> = None;

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

            w.add_document(doc)
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

    // Set HuggingFace cache directory if specified
    if let Some(cache_dir) = &args.cache_dir {
        std::fs::create_dir_all(cache_dir)?;
        // SAFETY: We set this env var before any threads are spawned
        unsafe { std::env::set_var("HF_HOME", cache_dir) };
        info!("HuggingFace cache directory: {:?}", cache_dir);
    }

    // Create data directory if needed
    std::fs::create_dir_all(&args.data_dir)?;

    let addr: SocketAddr = args.addr.parse()?;

    let num_indexing_threads = args.indexing_threads.unwrap_or_else(num_cpus::get);

    let config = IndexConfig {
        max_indexing_memory_bytes: args.max_indexing_memory_mb * 1024 * 1024,
        num_indexing_threads,
        ..Default::default()
    };

    let registry = Arc::new(IndexRegistry::new(args.data_dir.clone(), config));

    let search_service = SearchServiceImpl {
        registry: Arc::clone(&registry),
    };

    let index_service = IndexServiceImpl {
        registry: Arc::clone(&registry),
    };

    info!("Starting Hermes server on {}", addr);
    info!("Data directory: {:?}", args.data_dir);
    info!("Max indexing memory: {} MB", args.max_indexing_memory_mb);
    info!("Indexing threads: {}", num_indexing_threads);

    // 256 MB limit for large batch index operations
    const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

    Server::builder()
        .add_service(
            SearchServiceServer::new(search_service)
                .max_decoding_message_size(MAX_MESSAGE_SIZE)
                .max_encoding_message_size(MAX_MESSAGE_SIZE)
                .accept_compressed(CompressionEncoding::Gzip)
                .send_compressed(CompressionEncoding::Gzip),
        )
        .add_service(
            IndexServiceServer::new(index_service)
                .max_decoding_message_size(MAX_MESSAGE_SIZE)
                .max_encoding_message_size(MAX_MESSAGE_SIZE)
                .accept_compressed(CompressionEncoding::Gzip)
                .send_compressed(CompressionEncoding::Gzip),
        )
        .serve(addr)
        .await?;

    Ok(())
}
