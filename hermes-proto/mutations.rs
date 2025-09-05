// Shared Rust envelope validation, included beside the generated bindings in
// server and broker. Keep limits here so routing cannot bypass server bounds.
impl DeleteDocumentsRequest {
    pub fn validate_limits(&self) -> Result<(), tonic::Status> {
        if self.primary_keys.len() > 100_000
            || self
                .primary_keys
                .iter()
                .try_fold(0usize, |total, key| total.checked_add(key.len()))
                .is_none_or(|bytes| bytes > 8 * 1024 * 1024)
        {
            return Err(tonic::Status::resource_exhausted(
                "deletion request exceeds 100000 keys or 8 MiB of key bytes",
            ));
        }
        Ok(())
    }
}

impl UpsertDocumentsRequest {
    pub fn validate_limits(&self) -> Result<(), tonic::Status> {
        if self.documents.len() > 1_000 || prost::Message::encoded_len(self) > 32 * 1024 * 1024 {
            return Err(tonic::Status::resource_exhausted(
                "upsert request exceeds 1000 documents or 32 MiB encoded bytes",
            ));
        }
        Ok(())
    }
}
