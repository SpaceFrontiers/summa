//! Borrow external IDs from the existing fast-field reader until serialization.
use serde::Serialize;

#[derive(Serialize)]
#[serde(untagged)]
pub(super) enum SearchResponse<'a> {
    // Field declaration order preserves the original JSON object's sorted keys.
    Count { docs: [(); 0], found: u64 },
    Ranked { docs: Vec<Hit<'a>> },
}

#[derive(Serialize)]
pub(super) struct Hit<'a> {
    pub id: &'a str,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn borrowed_ids_preserve_json_bytes_order_escaping_and_count_shape() {
        for ids in [
            vec![],
            vec!["", "quote\"slash\\", "雪\n\t\u{0000}", "same", "same"],
        ] {
            let old = json!({"docs": ids.iter().map(|id| json!({"id":id})).collect::<Vec<_>>()});
            let new = SearchResponse::Ranked {
                docs: ids.iter().map(|id| Hit { id }).collect(),
            };
            assert_eq!(
                serde_json::to_vec(&new).unwrap(),
                serde_json::to_vec(&old).unwrap()
            );
        }
        for found in [0, 1, u64::MAX] {
            let old = json!({"found":found,"docs":[]});
            let new = SearchResponse::Count { docs: [], found };
            assert_eq!(
                serde_json::to_vec(&new).unwrap(),
                serde_json::to_vec(&old).unwrap()
            );
        }
    }
}
