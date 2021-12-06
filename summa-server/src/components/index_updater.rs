use std::sync::Arc;
use std::time::Duration;
use rdkafka::error::{KafkaError, RDKafkaErrorCode};

use summa_core::components::{IndexWriterHolder, SummaSegmentAttributes};
use summa_core::configs::{ApplicationConfig, ConfigProxy};
use summa_core::utils::thread_handler::ThreadHandler;
use tantivy::directory::MmapDirectory;
use tantivy::{Index, SegmentId, SegmentMeta};
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};
use tokio::time;
use tracing::{info, info_span, instrument, warn, Instrument};
use crate::components::consumer_manager::process_message;

use crate::errors::{Error, SummaServerResult};

pub struct IndexUpdater {
    autocommit_thread: Option<ThreadHandler<SummaServerResult<()>>>,
    application_config: Arc<dyn ConfigProxy<ApplicationConfig>>,
    inner_index_updater: Arc<RwLock<InnerIndexUpdater>>,
}

impl IndexUpdater {
    /// Creates new `IndexUpdater`
    pub(super) async fn new(
        application_config: Arc<dyn ConfigProxy<ApplicationConfig>>,
        index: Index,
        index_writer_holder: Arc<RwLock<IndexWriterHolder>>,
        index_name: &str,
    ) -> SummaServerResult<IndexUpdater> {
        let inner_index_updater = InnerIndexUpdater {
            application_config: application_config.clone(),
            index,
            index_name: index_name.to_owned(),
            index_writer_holder,
        };
        let mut index_updater = IndexUpdater {
            autocommit_thread: None,
            application_config: application_config.clone(),
            inner_index_updater: Arc::new(RwLock::new(inner_index_updater)),
        };

        index_updater.start_autocommit_thread().await;
        Ok(index_updater)
    }

    pub async fn read(&self) -> RwLockReadGuard<'_, InnerIndexUpdater> {
        self.inner_index_updater.read().await
    }

    pub async fn write(&self) -> RwLockWriteGuard<'_, InnerIndexUpdater> {
        self.inner_index_updater.write().await
    }

    pub fn try_write(&self) -> Result<RwLockWriteGuard<'_, InnerIndexUpdater>, TryLockError> {
        self.inner_index_updater.try_write()
    }

    pub fn try_read(&self) -> Result<RwLockReadGuard<'_, InnerIndexUpdater>, TryLockError> {
        self.inner_index_updater.try_read()
    }

    async fn setup_autocommit_thread(&mut self) {
        let inner_index_updater = self.inner_index_updater.write().await;
        let index_name = inner_index_updater.index_name.clone();
        let interval_ms = match self.application_config.read().await.get().autocommit_interval_ms {
            Some(interval_ms) => interval_ms,
            None => return,
        };

        let (shutdown_trigger, mut shutdown_tripwire) = async_broadcast::broadcast(1);
        let mut tick_task = time::interval(Duration::from_millis(interval_ms));
        let inner_index_updater = self.inner_index_updater.clone();

        self.autocommit_thread = Some(ThreadHandler::new(
            tokio::spawn(
                async move {
                    info!(action = "spawning_autocommit_thread", interval_ms = interval_ms);
                    // The first tick ticks immediately so we skip it
                    tick_task.tick().await;
                    loop {
                        tokio::select! {
                            _ = tick_task.tick() => {
                                info!(action = "autocommit_thread_tick");
                                if let Ok(inner_index_updater) = inner_index_updater.try_read() {
                                    if let Err(error) = inner_index_updater.commit(None).await {
                                        warn!(error = ?error);
                                    }
                                }

                            }
                            _ = &mut shutdown_tripwire.recv() => {
                                info!(action = "shutdown_autocommit_thread");
                                break;
                            }
                        }
                    }
                    Ok(())
                }
                .instrument(info_span!(parent: None, "autocommit_thread", index_name = ?index_name)),
            ),
            shutdown_trigger,
        ));
    }

    /// Starts all consumers
    #[instrument(skip(self))]
    pub async fn start_autocommit_thread(&mut self) {
        self.setup_autocommit_thread().await;
    }

    /// Stops all consumers
    #[instrument(skip(self))]
    pub async fn stop_autocommit_thread(&mut self) -> SummaServerResult<()> {
        if let Some(autocommit_thread) = self.autocommit_thread.take() {
            autocommit_thread.stop().await??;
        }
        Ok(())
    }

    /// Stops consumers, commits Tantivy and Kafka offsets
    #[instrument(skip(self))]
    pub async fn stop_updates_and_commit(self) -> SummaServerResult<()> {
        let inner_index_updater = self.read().await;
        inner_index_updater.stop_consumers().await?;
        inner_index_updater.commit_index(None).await?;
        inner_index_updater.commit_offsets().await?;
        Ok(())
    }
}

/// Index updating through consumers and via direct invocation
pub struct InnerIndexUpdater {
    application_config: Arc<dyn ConfigProxy<ApplicationConfig>>,
    index: Index,
    index_name: String,
    index_writer_holder: Arc<RwLock<IndexWriterHolder>>,
}

impl InnerIndexUpdater {
    /// Tantivy `Index`
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Mutable Tantivy `Index`
    pub fn index_mut(&mut self) -> &mut Index {
        &mut self.index
    }

    /// Commits Kafka offsets
    #[instrument(skip(self))]
    pub(super) async fn commit_offsets(&self) -> SummaServerResult<()> {
        for consumer in &self.consumers {
            let commit_result = consumer.commit().await;
            if let Err(Error::Kafka(KafkaError::ConsumerCommit(RDKafkaErrorCode::AssignmentLost))) = commit_result {
                let schema = self.index.schema();
                let index_writer_holder = self.index_writer_holder.clone().read_owned().await;
                consumer.start(move |message| process_message(&index_writer_holder, &schema, message)).await;
            }
        }
        info!(action = "committed_offsets");
        Ok(())
    }

    /// Commits index
    #[instrument(skip(self))]
    pub(super) async fn commit_index(&self, payload: Option<String>) -> SummaServerResult<()> {
        Ok(self.index_writer_holder.write().await.commit(payload).await?)
    }

    /// Commits all
    #[instrument(skip(self))]
    pub async fn commit(&self, payload: Option<String>) -> SummaServerResult<()> {
        self.stop_consumers().await?;
        self.commit_index(payload).await?;
        self.commit_offsets().await?;
        self.start_consumers().await;
        Ok(())
    }
}
