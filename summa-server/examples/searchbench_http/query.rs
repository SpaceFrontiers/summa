//! Translate the declared benchmark envelope into existing core queries.

use anyhow::{Context, Result, bail};
use serde_json::Value;
use summa_core::dsl::{Field, QueryLanguageParser};
use summa_core::query::{PhraseQuery, Query, RegexQuery, WildcardQuery};
use summa_core::tokenizer::{BoxedTokenizer, Purpose};

pub struct Request {
    pub query: Box<dyn Query>,
    pub limit: usize,
}

pub struct Envelope {
    text: String,
    class: String,
    limit: usize,
}

impl Envelope {
    pub fn from_value(mut value: Value) -> Result<Self> {
        let text = value
            .get("query")
            .and_then(Value::as_str)
            .context("missing query")?;
        let _class = value
            .get("class")
            .and_then(Value::as_str)
            .context("missing class")?;
        if text.len() > 4096 {
            bail!("query exceeds 4096 bytes");
        }
        let limit = value
            .get("limit")
            .and_then(Value::as_u64)
            .context("missing limit")?;
        if !matches!(limit, 0 | 10 | 100) {
            bail!("limit must be 0 (exact count), 10, or 100");
        }
        if text.split_whitespace().count() > summa_core::query::MAX_QUERY_TERMS {
            bail!("too many query terms");
        }
        // IndexMut clones the key; borrowed lookups keep conversion allocation-free.
        let Value::String(text) = value.get_mut("query").expect("validated query").take() else {
            unreachable!()
        };
        let Value::String(class) = value.get_mut("class").expect("validated class").take() else {
            unreachable!()
        };
        Ok(Self {
            text,
            class,
            limit: limit as usize,
        })
    }

    pub fn parse(
        self,
        parser: &QueryLanguageParser,
        field: Field,
        tokenizer: &BoxedTokenizer,
    ) -> Result<Request> {
        let text = self.text.as_str();
        let query: Box<dyn Query> = match self.class.as_str() {
            "regex" => Box::new(RegexQuery::new(field, text)?),
            "wildcard" | "wildcard_scan" | "wildcard_lead" => {
                Box::new(WildcardQuery::text(field, text)?)
            }
            "high_sloppy_phrase" | "med_sloppy_phrase" | "low_sloppy_phrase" => {
                let quoted = text.strip_suffix("~4").context("expected slop 4")?;
                let phrase: String =
                    serde_json::from_str(quoted).context("expected quoted phrase")?;
                let tokens = tokenizer.tokenize_with(&phrase, None, Purpose::Exact);
                if tokens.len() > summa_core::query::MAX_QUERY_TERMS {
                    bail!("too many phrase terms");
                }
                Box::new(
                    PhraseQuery::with_offsets(
                        field,
                        tokens
                            .into_iter()
                            .map(|token| (token.position, token.text.into_bytes()))
                            .collect(),
                    )
                    .with_slop(4),
                )
            }
            "high_term" | "med_term" | "low_term" | "and_high_high" | "and_high_med"
            | "and_high_low" | "or_high_high" | "or_high_med" | "or_high_low" | "high_phrase"
            | "med_phrase" | "low_phrase" | "prefix3" => {
                parser.parse_strict(text).map_err(anyhow::Error::msg)?
            }
            class => bail!("unsupported query class: {class}"),
        };
        Ok(Request {
            query,
            limit: self.limit,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Thread-local measurement excludes concurrently running tests. The allocator
    // forwards the original pointers/layouts unchanged to System.
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    thread_local! {
        static MEASURE: Cell<bool> = const { Cell::new(false) };
        static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    }
    struct MeasuredAllocator;
    fn record_allocation() {
        let _ = MEASURE.try_with(|enabled| {
            if enabled.get() {
                ALLOCATIONS.with(|count| count.set(count.get() + 1));
            }
        });
    }
    // SAFETY: all allocation operations delegate to System with unchanged arguments.
    unsafe impl GlobalAlloc for MeasuredAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record_allocation();
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer, layout) }
        }
        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            record_allocation();
            unsafe { System.realloc(pointer, layout, size) }
        }
    }
    #[global_allocator]
    static ALLOCATOR: MeasuredAllocator = MeasuredAllocator;

    #[test]
    fn envelope_conversion_allocates_no_buffers() {
        let value = json!({"query": "alpha beta", "class": "and_high_high", "limit": 10});
        ALLOCATIONS.set(0);
        MEASURE.set(true);
        let envelope = Envelope::from_value(value);
        MEASURE.set(false);
        let allocations = ALLOCATIONS.get();
        assert_eq!(envelope.unwrap().text, "alpha beta");
        assert_eq!(allocations, 0);
    }

    #[test]
    fn envelope_moves_owned_string_buffers() {
        let value = json!({"query": "alpha beta", "class": "and_high_high", "limit": 10});
        let query = value["query"].as_str().unwrap().as_ptr();
        let class = value["class"].as_str().unwrap().as_ptr();
        let envelope = Envelope::from_value(value).unwrap();
        assert_eq!(envelope.text.as_ptr(), query);
        assert_eq!(envelope.class.as_ptr(), class);
        assert_eq!(envelope.limit, 10);
    }

    #[test]
    fn envelope_preserves_validation_order_and_limits() {
        for (value, error) in [
            (json!(null), "missing query"),
            (json!({"query": 2, "class": false}), "missing query"),
            (json!({"query": "a".repeat(4097)}), "missing class"),
            (
                json!({"query": "a".repeat(4097), "class": "high_term"}),
                "query exceeds 4096 bytes",
            ),
            (
                json!({"query": "a", "class": "high_term", "limit": -1}),
                "missing limit",
            ),
            (
                json!({"query": "a", "class": "high_term", "limit": 11}),
                "limit must be 0 (exact count), 10, or 100",
            ),
            (
                json!({"query": "a ".repeat(summa_core::query::MAX_QUERY_TERMS + 1), "class": "high_term", "limit": 0}),
                "too many query terms",
            ),
        ] {
            assert_eq!(
                Envelope::from_value(value).err().unwrap().to_string(),
                error
            );
        }
    }
}
