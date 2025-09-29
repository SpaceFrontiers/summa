//! Common word pairs answer exact two-word phrases exactly as positions do
//! (`docs/common-word-pairs.md`).

use crate::directories::RamDirectory;
use crate::dsl::{Document, PositionMode, SchemaBuilder};
use crate::query::{
    CountCollector, PhraseQuery, Query, RegexQuery, TermQuery, WildcardQuery, collect_segment,
};
use crate::{Index, IndexConfig, IndexWriter};

const VOCABULARY: [&str; 9] = ["a", "the", "of", "run", "runs", "running", "x", "y", "of"];
const COMMON: [&str; 5] = ["a", "the", "of", "run", "runs"];

struct Case {
    tokenizer: &'static str,
    positions: PositionMode,
    reorder: bool,
    /// Deletions route exact counts through the pair's postings instead of
    /// its document frequency.
    delete: bool,
}

async fn build(case: &Case, common: bool, seed: u64) -> Index<RamDirectory> {
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field_with_tokenizer("id", true, true, "raw");
    schema.set_primary_key(id);
    let body = schema.add_text_field_with_tokenizer("body", true, false, case.tokenizer);
    schema.set_positions(body, case.positions);
    schema.set_multi(body, true);
    schema.set_reorder(body, case.reorder);
    if common {
        schema.set_common_grams(body, COMMON.iter().map(|word| word.to_string()).collect());
    }
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for segment in 0..2 {
        for row in 0..600 {
            let mut doc = Document::new();
            doc.add_text(id, format!("{segment}-{row}"));
            // Some documents carry two values: in TokenPosition mode their
            // positions overlap, and phrases may match across them.
            for _ in 0..1 + (next() % 5 == 0) as usize {
                let length = next() % 24;
                let text: Vec<&str> = (0..length)
                    .map(|_| VOCABULARY[(next() % VOCABULARY.len() as u64) as usize])
                    .collect();
                doc.add_text(body, text.join(" "));
            }
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.force_merge().await.unwrap();
    if case.reorder {
        writer.reorder().await.unwrap();
    }
    if case.delete {
        for key in ["0-3", "0-77", "1-5", "1-599"] {
            writer.delete_primary_key(key).unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.shutdown().await.unwrap();
    Index::open(dir, config).await.unwrap()
}

async fn count(index: &Index<RamDirectory>, query: &dyn Query) -> u64 {
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let mut total = 0;
    for segment in searcher.segment_readers() {
        let mut collector = CountCollector::new();
        collect_segment(segment, query, &mut collector)
            .await
            .unwrap();
        total += collector.count();
    }
    total
}

async fn ranked(index: &Index<RamDirectory>, query: &dyn Query, limit: usize) -> Vec<(u32, u32)> {
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    assert_eq!(searcher.num_segments(), 1);
    searcher
        .search(query, limit)
        .await
        .unwrap()
        .iter()
        .map(|hit| (hit.doc_id, hit.score.to_bits()))
        .collect()
}

/// Terms an accept-everything expansion of the whole field returns.
async fn whole_field(index: &Index<RamDirectory>, field: crate::Field) -> usize {
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let prefixes = [Vec::new()];
    let lookup = crate::segment::reader::TermDictionaryLookup::prefixes(&prefixes);
    searcher.segment_readers()[0]
        .get_matching_postings(field, lookup, "test", usize::MAX, |_| true)
        .await
        .unwrap()
        .len()
}

#[tokio::test]
async fn common_word_pairs_answer_two_word_phrases_like_positions() {
    let cases = [
        Case {
            tokenizer: "default",
            positions: PositionMode::TokenPosition,
            reorder: false,
            delete: false,
        },
        Case {
            tokenizer: "lex(default: en, stem: snowball, variants: true)",
            positions: PositionMode::Full,
            reorder: false,
            delete: true,
        },
        Case {
            tokenizer: "lex(default: en, stem: snowball, variants: true)",
            positions: PositionMode::TokenPosition,
            reorder: true,
            delete: true,
        },
    ];
    for (seed, case) in cases.iter().enumerate() {
        let paired = build(case, true, seed as u64 + 1).await;
        let plain = build(case, false, seed as u64 + 1).await;
        let body = paired.schema().get_field("body").unwrap();
        let label = format!(
            "{} {:?} reorder={} delete={}",
            case.tokenizer, case.positions, case.reorder, case.delete
        );

        let reader = paired.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        let segment = &searcher.segment_readers()[0];
        let pair = crate::structures::word_pairs::word_pair_term(b"of", b"the");
        assert!(
            segment.text_doc_freq(body, &pair).await.unwrap() > 0,
            "{label}: pairs are indexed"
        );

        // On a reordered field a ranked pair keeps the logical term plan,
        // which stops after `k` stable IDs; complete streams go physical.
        if case.reorder {
            let phrase = PhraseQuery::new(body, vec![b"of".to_vec(), b"the".to_vec()]);
            assert_eq!(phrase.physical_text_field(segment, false), None, "{label}");
            assert_eq!(
                phrase.physical_text_field(segment, true),
                Some(body),
                "{label}"
            );
            let plain_reader = plain.reader().await.unwrap();
            let plain_searcher = plain_reader.searcher().await.unwrap();
            let plain_segment = &plain_searcher.segment_readers()[0];
            assert_eq!(phrase.physical_text_field(plain_segment, false), Some(body));
        }

        let mut words: Vec<&str> = VOCABULARY.to_vec();
        words.dedup();
        for first in &words {
            for second in &words {
                let phrase = PhraseQuery::new(
                    body,
                    vec![first.as_bytes().to_vec(), second.as_bytes().to_vec()],
                );
                let case_label = format!("{label} \"{first} {second}\"");
                assert_eq!(
                    count(&paired, &phrase).await,
                    count(&plain, &phrase).await,
                    "{case_label}: count"
                );
                for limit in [1, 10, 1000] {
                    assert_eq!(
                        ranked(&paired, &phrase, limit).await,
                        ranked(&plain, &phrase, limit).await,
                        "{case_label}: top {limit}"
                    );
                }
            }
        }
        // The paired index answers from the pair's postings, never positions.
        #[cfg(feature = "query-diagnostics")]
        {
            let phrase = PhraseQuery::new(body, vec![b"of".to_vec(), b"the".to_vec()]);
            let (_, work) = crate::search_diagnostics::capture(ranked(&paired, &phrase, 10)).await;
            assert_eq!(work.phrase_confirmations, 0, "{label}");
            assert_eq!(work.position_reads, 0, "{label}");
            let (_, work) = crate::search_diagnostics::capture(ranked(&plain, &phrase, 10)).await;
            assert!(work.phrase_confirmations > 0, "{label}");
        }
        // Dictionary expansions never see pair terms, even over a whole field.
        assert_eq!(
            whole_field(&paired, body).await,
            whole_field(&plain, body).await,
            "{label}: whole-field expansion"
        );
        let expansions: Vec<Box<dyn Query>> = vec![
            Box::new(RegexQuery::new(body, ".*").unwrap()),
            Box::new(WildcardQuery::new(body, "*e").unwrap()),
            Box::new(TermQuery::new(body, b"the".to_vec())),
        ];
        for query in &expansions {
            assert_eq!(
                count(&paired, query.as_ref()).await,
                count(&plain, query.as_ref()).await,
                "{label}: {query}"
            );
        }
    }
}

#[tokio::test]
async fn common_grams_are_validated_at_schema_admission_and_by_the_tokenizer() {
    let admit = |configure: &dyn Fn(&mut SchemaBuilder)| {
        let mut schema = SchemaBuilder::default();
        configure(&mut schema);
        schema.build().validate()
    };
    let rejected = [
        admit(&|schema| {
            let body = schema.add_text_field("body", true, false);
            schema.set_common_grams(body, vec!["the".into()]);
        }),
        admit(&|schema| {
            let body = schema.add_u64_field("n", true, false);
            schema.set_common_grams(body, vec!["the".into()]);
        }),
        admit(&|schema| {
            let body = schema.add_text_field("body", true, false);
            schema.set_positions(body, PositionMode::Full);
            schema.set_common_grams(body, vec!["the".into(), "the".into()]);
        }),
        admit(&|schema| {
            let body = schema.add_text_field("body", true, false);
            schema.set_positions(body, PositionMode::Ordinal);
            schema.set_common_grams(body, vec!["the".into()]);
        }),
    ];
    for result in rejected {
        assert!(matches!(result, Err(crate::Error::Schema(_))), "{result:?}");
    }

    // A word the tokenizer never produces as itself is refused loudly.
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("body", true, false);
    schema.set_positions(body, PositionMode::TokenPosition);
    schema.set_common_grams(body, vec!["The".into()]);
    let builder = crate::segment::SegmentBuilder::new(
        std::sync::Arc::new(schema.build()),
        Default::default(),
    );
    assert!(matches!(builder, Err(crate::Error::Schema(_))));
}
