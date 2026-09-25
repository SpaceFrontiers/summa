use super::BooleanScorer;
use crate::query::docset::{DOC_WINDOW_SIZE, DOC_WINDOW_WORDS, DocWindow};
use crate::query::{DocSet, Scorer};
use crate::structures::TERMINATED;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct WindowedDocs {
    docs: Vec<u32>,
    offset: usize,
    windows: Arc<AtomicUsize>,
}
impl DocSet for WindowedDocs {
    fn doc(&self) -> u32 {
        self.docs.get(self.offset).copied().unwrap_or(TERMINATED)
    }
    fn advance(&mut self) -> u32 {
        self.offset = (self.offset + 1).min(self.docs.len());
        self.doc()
    }
    fn seek(&mut self, target: u32) -> u32 {
        self.offset += self.docs[self.offset..].partition_point(|&doc| doc < target);
        self.doc()
    }
    fn size_hint(&self) -> u32 {
        self.docs.len() as u32
    }
    fn supports_doc_windows(&self) -> bool {
        true
    }
    fn fill_doc_window(&mut self, base: u32, bits: &mut DocWindow) {
        self.windows.fetch_add(1, Ordering::Relaxed);
        bits.fill(0);
        self.seek(base);
        let end = base.saturating_add(DOC_WINDOW_SIZE);
        let stop = self.offset + self.docs[self.offset..].partition_point(|&doc| doc < end);
        for &doc in &self.docs[self.offset..stop] {
            let offset = (doc - base) as usize;
            bits[offset / 64] |= 1 << (offset % 64);
        }
        self.offset = stop;
    }
}
impl Scorer for WindowedDocs {
    fn score(&self) -> f32 {
        (self.doc() % 17) as f32 * 0.5
    }
}
fn build(a: &[u32], b: &[u32]) -> (BooleanScorer<'static>, [Arc<AtomicUsize>; 2]) {
    let calls = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
    let mut scorer = BooleanScorer {
        must: vec![
            Box::new(WindowedDocs {
                docs: a.to_vec(),
                offset: 0,
                windows: calls[0].clone(),
            }),
            Box::new(WindowedDocs {
                docs: b.to_vec(),
                offset: 0,
                windows: calls[1].clone(),
            }),
        ],
        should: Vec::new(),
        must_not: Vec::new(),
        current_doc: 0,
        lead: 0,
        doc_limit: 0,
    };
    scorer.initialize();
    (scorer, calls)
}

#[test]
fn required_alignment_preserves_all_driver_orders_optional_scores_and_exclusions() {
    let make = |docs: Vec<u32>| -> Box<dyn Scorer> {
        Box::new(WindowedDocs {
            docs,
            offset: 0,
            windows: Arc::new(AtomicUsize::new(0)),
        })
    };
    for offset in [0, u32::MAX - 13000] {
        for rarest in 0..3 {
            let mut steps = [5, 11, 19];
            steps.swap(rarest, 2);
            for target in [0, offset + 31, offset + 4095, offset + 8193, TERMINATED] {
                let mut scorer = BooleanScorer {
                    must: steps
                        .iter()
                        .map(|step| {
                            make((0..12345).step_by(*step).map(|doc| offset + doc).collect())
                        })
                        .collect(),
                    should: vec![make(
                        (0..12345).step_by(3).map(|doc| offset + doc).collect(),
                    )],
                    must_not: vec![make(
                        (0..12345).step_by(7).map(|doc| offset + doc).collect(),
                    )],
                    current_doc: 0,
                    lead: 0,
                    doc_limit: 0,
                };
                scorer.initialize();
                assert_eq!(scorer.lead, rarest);
                scorer.seek(target);
                let expected: Vec<_> = (0..12345)
                    .filter(|doc| {
                        steps
                            .iter()
                            .all(|step| (*doc as usize).is_multiple_of(*step))
                            && doc % 7 != 0
                            && offset + doc >= target
                    })
                    .collect();
                for doc in expected {
                    assert_eq!(scorer.doc(), offset + doc);
                    assert_eq!(scorer.seek(0), offset + doc);
                    let value = ((offset + doc) % 17) as f32 * 0.5;
                    let mut score = value + value + value;
                    if doc % 3 == 0 {
                        score += value;
                    }
                    assert_eq!(scorer.score().to_bits(), score.to_bits());
                    scorer.advance();
                }
                assert_eq!(scorer.doc(), TERMINATED);
                assert_eq!(scorer.seek(0), TERMINATED);
                assert_eq!(scorer.advance(), TERMINATED);
            }
        }
    }
}

#[test]
fn conjunction_windows_probe_sparse_candidates_and_batch_dense_candidates() {
    let common: Vec<_> = (0..12345).collect();
    for lead in [
        (0..12345).step_by(2).collect::<Vec<_>>(),
        vec![0, 2000, 4095, 4100, 8000, 12000],
    ] {
        let (mut scalar, _) = build(&common, &lead);
        let (mut batched, calls) = build(&common, &lead);
        assert!(batched.supports_doc_windows());
        assert!(!batched.supports_score_windows());
        while batched.doc() != TERMINATED {
            let base = batched.doc();
            let end = base.saturating_add(DOC_WINDOW_SIZE);
            let mut bits = [u64::MAX; DOC_WINDOW_WORDS];
            batched.fill_doc_window(base, &mut bits);
            let mut expected = Vec::new();
            while scalar.doc() < end {
                expected.push(scalar.doc());
                scalar.advance();
            }
            let mut actual = Vec::new();
            for (i, &word) in bits.iter().enumerate() {
                for b in 0..64 {
                    if word & (1 << b) != 0 {
                        actual.push(base + i as u32 * 64 + b);
                    }
                }
            }
            assert_eq!(actual, expected);
            assert_eq!(batched.doc(), scalar.doc());
            if scalar.doc() != TERMINATED {
                assert_eq!(batched.score().to_bits(), scalar.score().to_bits());
            }
        }
        assert!(calls[1].load(Ordering::Relaxed) > 0);
        assert_eq!(
            calls[0].load(Ordering::Relaxed) > 0,
            lead.len() > DOC_WINDOW_WORDS
        );
    }
}

#[test]
fn conjunction_windows_preserve_empty_and_high_document_id_boundaries() {
    for (a, b) in [
        (vec![], vec![0]),
        (
            (0..16000).step_by(2).collect(),
            (1..16000).step_by(2).collect(),
        ),
        (
            (u32::MAX - 9000..u32::MAX).step_by(2).collect(),
            (u32::MAX - 9000..u32::MAX).step_by(3).collect(),
        ),
    ] {
        for start in [0, 4095, u32::MAX - 4097, u32::MAX - 1, TERMINATED] {
            let (mut scalar, _) = build(&a, &b);
            let (mut batched, _) = build(&a, &b);
            scalar.seek(start);
            batched.seek(start);
            while batched.doc() != TERMINATED {
                let base = batched.doc();
                let end = base.saturating_add(DOC_WINDOW_SIZE);
                let mut bits = [u64::MAX; DOC_WINDOW_WORDS];
                batched.fill_doc_window(base, &mut bits);
                for (i, &word) in bits.iter().enumerate() {
                    for b in 0..64 {
                        if word & (1 << b) != 0 {
                            assert_eq!(base + i as u32 * 64 + b, scalar.doc());
                            scalar.advance();
                        }
                    }
                }
                assert!(scalar.doc() >= end);
                assert_eq!(batched.doc(), scalar.doc());
            }
            let mut bits = [u64::MAX; DOC_WINDOW_WORDS];
            batched.fill_doc_window(TERMINATED, &mut bits);
            assert!(bits.iter().all(|&word| word == 0));
            assert_eq!(batched.seek(0), TERMINATED);
        }
    }
}

#[test]
fn sparse_conjunctions_avoid_bitmap_setup_and_dense_conjunctions_keep_windows() {
    let common: Vec<_> = (0..12_345).collect();
    let sparse: Vec<_> = (0..12_345).step_by(1000).collect();
    let dense: Vec<_> = (0..12_345).step_by(2).collect();
    let (mut sparse, _) = build(&sparse, &common);
    sparse.doc_limit = 12_345;
    assert!(!sparse.supports_doc_windows());
    let (mut dense, _) = build(&dense, &common);
    dense.doc_limit = 12_345;
    assert!(dense.supports_doc_windows());
    // The multiplication uses the full document-ID range without overflow.
    dense.doc_limit = u32::MAX;
    assert!(!dense.supports_doc_windows());
}
