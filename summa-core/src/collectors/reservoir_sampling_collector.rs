use rand::rngs::SmallRng;
use rand::{Rng, RngCore, SeedableRng};
use tantivy::collector::Collector;
use tantivy::collector::SegmentCollector;
use tantivy::{DocAddress, DocId, Score, SegmentOrdinal, SegmentReader};

/// `ReservoirSampling` collector collects `k` random documents using `O(k)` memory
///
/// ```rust
/// use summa_core::collectors::ReservoirSampling;
/// use summa_core::configs::core::QueryParserConfig;
/// use summa_core::components::{MorphologyManager, QueryParser};
/// use tantivy::collector::Count;
/// use tantivy::schema::{Schema, TEXT};
/// use tantivy::{doc, Index};
///
/// let mut schema_builder = Schema::builder();
/// let title = schema_builder.add_text_field("title", TEXT);
/// let schema = schema_builder.build();
/// let index = Index::create_in_ram(schema);
///
/// let mut index_writer = index.writer(3_000_000).unwrap();
/// index_writer.add_document(doc!(title => "The Name of the Wind")).unwrap();
/// index_writer.add_document(doc!(title => "The Diary of Muadib")).unwrap();
/// index_writer.add_document(doc!(title => "A Dairy Cow")).unwrap();
/// index_writer.add_document(doc!(title => "The Diary of a Young Girl")).unwrap();
/// assert!(index_writer.commit().is_ok());
///
/// let reader = index.reader().unwrap();
/// let searcher = reader.searcher();
///
/// // Here comes the important part
/// let query_parser = QueryParser::for_index(&index, QueryParserConfig::from_default_fields(vec!["title".to_string()]), &MorphologyManager::default()).unwrap();
/// let query = query_parser.parse_query("diary").unwrap();
/// let documents = searcher.search(&query, &ReservoirSampling::with_limit(2)).unwrap();
///
/// assert_eq!(documents.len(), 2);
/// ```
pub struct ReservoirSampling {
    limit: usize,
}

/// Implements [Algorithm R](https://en.wikipedia.org/wiki/Reservoir_sampling#Simple_algorithm)
/// for weighted sampling from the downstream `Fruit`. It uses `O(k)` memory and has `O(n)` time complexity.
impl ReservoirSampling {
    pub fn with_limit(limit: usize) -> ReservoirSampling {
        ReservoirSampling { limit }
    }
}

impl Collector for ReservoirSampling {
    type Fruit = Vec<DocAddress>;

    type Child = SegmentReservoirSamplingCollector;

    fn for_segment(&self, segment_ord: SegmentOrdinal, _: &SegmentReader) -> tantivy::Result<SegmentReservoirSamplingCollector> {
        Ok(SegmentReservoirSamplingCollector::new(segment_ord, self.limit))
    }

    fn requires_scoring(&self) -> bool {
        false
    }

    fn merge_fruits(&self, segment_docs_vec: Vec<(Vec<DocAddress>, usize)>) -> tantivy::Result<Vec<DocAddress>> {
        let mut total_reservoir = vec![];
        let mut seen_documents = 0;

        let mut rng = SmallRng::from_entropy();

        for (segment_docs, segment_size) in segment_docs_vec.iter().filter(|(_, segment_size)| *segment_size > 0) {
            // Tracking how much documents has been already taken from the current segment.
            // Required for the trick for maintaining fair distribution
            let mut taken_from_current_segment = 0;

            seen_documents += segment_size;
            for doc in segment_docs {
                // Fill the reservoir initially
                if total_reservoir.len() < self.limit {
                    total_reservoir.push(*doc)
                } else {
                    // Trial if the current document from the current `segment_docs` should be taken and put into reservoir taking into
                    // account the global distribution of the documents
                    if (rng.next_u32() as usize) % seen_documents < *segment_size {
                        taken_from_current_segment += 1;
                        // The goal is to put the document from the current segment instead of documents from
                        // the document collected from previous iterations.
                        //
                        // For this purpose we are virtually splitting `total_reservoir` into two parts:
                        // - `total_reservoir[0; self.limit - taken_from_current_segment]`
                        // - `total_reservoir[self.limit - taken_from_current_segment; self.limit]`
                        //
                        // The first one contains previously collected documents and the second one contains document from the current segment.
                        let pivot_index = self.limit - taken_from_current_segment;
                        if pivot_index > 0 {
                            let position_to_swap = (rng.next_u32() as usize) % pivot_index;
                            total_reservoir.swap(pivot_index, position_to_swap);
                        }
                        total_reservoir[pivot_index] = *doc;
                    }
                }
            }
        }
        Ok(total_reservoir)
    }
}

pub struct SegmentReservoirSamplingCollector {
    segment_ord: u32,
    reservoir: Vec<DocAddress>,
    seen_segment_docs: usize,
    limit: usize,
    rng: SmallRng,
    next_element: usize,
    w: f64,
}

#[inline]
fn gd_gap<TRng: Rng>(w: f64, rng: &mut TRng) -> usize {
    (rng.gen::<f64>().ln() / (1.0 - w).ln()).floor() as usize + 1
}

#[inline]
fn w_mul<TRng: Rng>(limit: usize, rng: &mut TRng) -> f64 {
    (rng.gen::<f64>().ln() / limit as f64).exp()
}

/// Implements [Algorithm L](https://en.wikipedia.org/wiki/Reservoir_sampling#An_optimal_algorithm) for reservoir sampling of size `k` from `n` elements
/// found by the upstream `Query`
/// It uses `O(k)` memory and has `O(k *(1 + log(n / k)))` time complexity
impl SegmentReservoirSamplingCollector {
    pub fn new(segment_ord: u32, limit: usize) -> SegmentReservoirSamplingCollector {
        let mut rng = SmallRng::from_entropy();

        let w = 1f64 * w_mul(limit, &mut rng);
        let next_element = limit + gd_gap(w, &mut rng);

        SegmentReservoirSamplingCollector {
            segment_ord,
            reservoir: vec![],
            seen_segment_docs: 0,
            limit,
            rng,
            next_element,
            w,
        }
    }
}

impl SegmentCollector for SegmentReservoirSamplingCollector {
    type Fruit = (Vec<DocAddress>, usize);

    fn collect(&mut self, doc_id: DocId, _: Score) {
        self.seen_segment_docs += 1;
        if self.reservoir.len() < self.limit {
            // Initial filling of the reservoir
            self.reservoir.push(DocAddress::new(self.segment_ord, doc_id));
        } else if self.seen_segment_docs == self.next_element {
            self.reservoir[(self.rng.next_u32() as usize) % self.limit] = DocAddress::new(self.segment_ord, doc_id);
            self.w *= w_mul(self.limit, &mut self.rng);
            self.next_element += gd_gap(self.w, &mut self.rng);
        }
    }

    fn harvest(self) -> (Vec<DocAddress>, usize) {
        (self.reservoir, self.seen_segment_docs)
    }
}

#[cfg(test)]
mod tests {
    use tantivy::collector::Collector;

    use super::ReservoirSampling;

    #[test]
    fn test_count_collect_does_not_requires_scoring() {
        assert!(!ReservoirSampling::with_limit(0).requires_scoring());
    }

    #[test]
    fn test_border_cases() {
        assert!(!ReservoirSampling::with_limit(0).requires_scoring());
    }
}
