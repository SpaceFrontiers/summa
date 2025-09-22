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
        let limit_mib = if self.documents.len() == 1 { 200 } else { 32 };
        if self.documents.len() > 1_000
            || prost::Message::encoded_len(self) > limit_mib * 1024 * 1024
        {
            return Err(tonic::Status::resource_exhausted(format!(
                "upsert request exceeds 1000 documents or {limit_mib} MiB encoded bytes",
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod mutation_limits_tests {
    use super::*;
    use prost::Message;

    fn request_with_encoded_size(documents: usize, bytes: usize) -> UpsertDocumentsRequest {
        let mut request = UpsertDocumentsRequest {
            index_name: "large".into(),
            documents: vec![NamedDocument::default(); documents],
        };
        request.documents[0].fields.push(FieldEntry {
            name: "body".into(),
            value: Some(FieldValue {
                value: Some(field_value::Value::Text("x".repeat(bytes))),
            }),
        });
        let overhead = request.encoded_len() - bytes;
        let Some(field_value::Value::Text(text)) = request.documents[0].fields[0]
            .value
            .as_mut()
            .unwrap()
            .value
            .as_mut()
        else {
            unreachable!()
        };
        text.truncate(bytes - overhead);
        assert_eq!(request.encoded_len(), bytes);
        request
    }

    #[test]
    fn singleton_upsert_limit_counts_complete_envelope_and_preserves_batch_limit() {
        for (documents, limit) in [(1, 200 * 1024 * 1024), (2, 32 * 1024 * 1024)] {
            let mut request = request_with_encoded_size(documents, limit);
            request.validate_limits().unwrap();
            request.index_name.push('x');
            assert_eq!(
                request.validate_limits().unwrap_err().code(),
                tonic::Code::ResourceExhausted
            );
        }
        assert_eq!(
            UpsertDocumentsRequest {
                index_name: "large".into(),
                documents: vec![NamedDocument::default(); 1001],
            }
            .validate_limits()
            .unwrap_err()
            .code(),
            tonic::Code::ResourceExhausted
        );
    }
}
