use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, RwLock,
};

use std::time::{Duration, Instant};

use tantivy::query::QueryParser;
use tantivy::schema::{Document, Schema};
use tantivy::{Index, IndexReader, IndexWriter, Term};

use crate::ThreadHandler;

#[derive(Clone, Serialize, Deserialize)]
pub struct SchemaConfig {
    pub default_fields: Vec<String>,
    pub enabled: bool,
    pub key_field: String,
    pub multi_fields: Vec<String>,
    pub name: String,
    pub schema: Schema,
}

#[derive(Clone)]
pub struct SchemaEngine {
    pub document_receiver: crossbeam_channel::Receiver<Document>,
    pub document_sender: crossbeam_channel::Sender<Document>,
    pub index: Index,
    pub index_reader: IndexReader,
    pub index_writer: Arc<RwLock<IndexWriter>>,
    pub logger: slog::Logger,
    pub query_parser: QueryParser,
    pub schema_config: SchemaConfig,
}

impl SchemaEngine {
    pub fn start_auto_commit_thread(&self, timeout: Duration, wakeup_timeout: Duration) -> ThreadHandler {
        let running = Arc::new(AtomicBool::new(true));
        let running_in_thread = running.clone();

        let mut last_commit_time = Instant::now();
        let mut current_timeout = timeout.clone();

        let id_field = self.schema_config
            .schema
            .get_field(&self.schema_config.key_field)
            .unwrap();

        let document_receiver = self.document_receiver.clone();
        let index_writer = self.index_writer.clone();
        let logger = self.logger.clone();
        let schema_config_name = self.schema_config.name.clone();

        let join_handle = Box::new(std::thread::spawn(move || -> Result<(), crate::errors::Error> {
            info!(logger, "start";
                "action" => "start",
                "mode" => "auto_commit_thread",
                "schema" => &schema_config_name,
            );

            let mut upserted = 0;

            while running_in_thread.load(Ordering::Acquire) || !document_receiver.is_empty() {
                if let Ok(document) =
                    document_receiver.recv_timeout(std::cmp::min(current_timeout, wakeup_timeout))
                {
                    let index_writer = index_writer.read().unwrap();
                    index_writer.delete_term(Term::from_field_i64(
                        id_field,
                        document.get_first(id_field).unwrap().i64_value().unwrap(),
                    ));
                    index_writer.add_document(document);
                    upserted += 1;
                };

                let now = Instant::now();
                if last_commit_time + timeout <= now {
                    if upserted > 0 {
                        info!(logger, "auto_commit";
                            "action" => "auto_commit",
                            "mode" => "auto_commit_thread",
                            "schema" => &schema_config_name,
                            "upserted" => upserted,
                        );
                        index_writer.write().unwrap().commit()?;
                        upserted = 0;
                    }
                    last_commit_time = Instant::now();
                    current_timeout = timeout.clone();
                } else {
                    current_timeout = last_commit_time + timeout - now;
                };
            }
            info!(logger, "final_auto_commit";
                "action" => "final_auto_commit",
                "mode" => "auto_commit_thread",
                "left_documents" => document_receiver.len(),
                "schema" => &schema_config_name,
                "upserted" => upserted,
            );
            index_writer.write().unwrap().commit()?;
            info!(logger, "exit";
                "action" => "exit",
                "mode" => "auto_commit_thread",
                "schema" => &schema_config_name,
            );
            Ok(())
        }));
        ThreadHandler::new(join_handle, running)
    }

    pub fn commit(&self) -> Result<(), crate::errors::Error> {
        self.index_writer.write().unwrap().commit()?;
        Ok(())
    }
}