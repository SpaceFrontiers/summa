use std::fmt::Debug;

use async_trait::async_trait;
use summa_core::components::IndexWriterHolder;
use summa_proto::proto;
use tantivy::schema::Schema;
use tokio::sync::OwnedRwLockReadGuard;

use crate::SummaServerResult;

#[async_trait]
pub trait ConsumerThread: Send + Sync + Debug {
    fn consumer_name(&self) -> &str;
    async fn start(
        &self,
        index_writer_holder: OwnedRwLockReadGuard<IndexWriterHolder>,
        conflict_strategy: proto::ConflictStrategy,
        schema: Schema,
    ) -> SummaServerResult<()>;
    async fn stop(&self) -> SummaServerResult<()>;
    async fn commit(&self) -> SummaServerResult<()>;
    async fn on_create(&self) -> SummaServerResult<()>;
    async fn on_delete(&self) -> SummaServerResult<()>;
    fn config(&self) -> &crate::configs::consumer::Config;
}
