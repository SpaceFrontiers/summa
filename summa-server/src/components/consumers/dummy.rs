use std::fmt::Debug;

use async_trait::async_trait;
use summa_core::components::IndexWriterHolder;
use summa_proto::proto::ConflictStrategy;
use tantivy::schema::Schema;
use tokio::sync::OwnedRwLockReadGuard;

use crate::components::consumers::ConsumerThread;
use crate::configs::consumer::Config;
use crate::SummaServerResult;

#[derive(Debug)]
struct DummyConsumerThread;

#[async_trait]
impl ConsumerThread for DummyConsumerThread {
    fn consumer_name(&self) -> &str {
        unimplemented!()
    }

    async fn start(&self, _: OwnedRwLockReadGuard<IndexWriterHolder>, _: ConflictStrategy, _: Schema) -> SummaServerResult<()> {
        unimplemented!()
    }

    async fn stop(&self) -> SummaServerResult<()> {
        unimplemented!()
    }

    async fn commit(&self) -> SummaServerResult<()> {
        unimplemented!()
    }

    async fn on_create(&self) -> SummaServerResult<()> {
        unimplemented!()
    }

    async fn on_delete(&self) -> SummaServerResult<()> {
        unimplemented!()
    }

    fn config(&self) -> &Config {
        unimplemented!()
    }
}
