//! Bounded RPC translation and admission. The core writer owns row mutations.
use super::*;

// Conversion and staging share capacity. Detached blocking work retains its
// permit and writer guard through completion if the caller cancels.
static MUTATION_ADMISSION: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(4);

impl IndexServiceImpl {
    pub(super) async fn stage_deletions(
        &self,
        req: DeleteDocumentsRequest,
    ) -> Result<DocumentMutationResponse, Status> {
        req.validate_limits()?;
        let permit = MUTATION_ADMISSION.try_acquire().map_err(|_| {
            Status::resource_exhausted("document mutation capacity exhausted; retry later")
        })?;
        let index = self.registry.get_or_open_index(&req.index_name).await?;
        require_primary_key(&index.schema())?;
        let writer = self.registry.get_writer(&req.index_name).await?;
        // Cancellation while waiting for the writer cannot start staging.
        let mut writer = writer.write_owned().await;
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let mut response = DocumentMutationResponse::default();
            for (position, key) in req.primary_keys.into_iter().enumerate() {
                record(&mut response, position, writer.delete_primary_key(&key));
            }
            response
        })
        .await
        .map_err(|error| Status::internal(format!("deletion staging failed: {error}")))
    }

    pub(super) async fn stage_upserts(
        &self,
        req: UpsertDocumentsRequest,
    ) -> Result<DocumentMutationResponse, Status> {
        req.validate_limits()?;
        let permit = MUTATION_ADMISSION.try_acquire().map_err(|_| {
            Status::resource_exhausted("document mutation capacity exhausted; retry later")
        })?;
        let index = self.registry.get_or_open_index(&req.index_name).await?;
        let schema = index.schema();
        require_primary_key(&schema)?;
        let writer = self.registry.get_writer(&req.index_name).await?;
        let (documents, permit) = tokio::task::spawn_blocking(move || {
            let documents: Vec<_> = req
                .documents
                .into_iter()
                .map(|document| convert_proto_to_document(&document.fields, &schema))
                .collect();
            (documents, permit)
        })
        .await
        .map_err(|error| Status::internal(format!("upsert conversion failed: {error}")))?;
        let mut writer = writer.write_owned().await;
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let mut response = DocumentMutationResponse::default();
            for (position, document) in documents.into_iter().enumerate() {
                let result = document.map_err(|error| error.to_string()).and_then(|doc| {
                    writer
                        .upsert_document(doc)
                        .map_err(|error| error.to_string())
                });
                record(&mut response, position, result);
            }
            response
        })
        .await
        .map_err(|error| Status::internal(format!("upsert staging failed: {error}")))
    }
}

fn require_primary_key(schema: &hermes_core::Schema) -> Result<(), Status> {
    if schema.primary_field().is_none() {
        return Err(Status::failed_precondition(
            "document mutations require a primary-key schema",
        ));
    }
    Ok(())
}

fn record<E: std::fmt::Display>(
    response: &mut DocumentMutationResponse,
    position: usize,
    result: Result<(), E>,
) {
    match result {
        Ok(()) => response.accepted_count += 1,
        Err(error) => response.errors.push(DocumentError {
            index: position as u32,
            error: error.to_string(),
        }),
    }
}
