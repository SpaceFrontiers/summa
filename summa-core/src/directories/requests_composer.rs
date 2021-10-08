use crate::directories::chunk_generator::Chunk;
use std::iter::{Chain, Once};
use std::ops::Range;
use std::slice::Iter;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Request {
    first_chunk: Chunk,
    other_chunks: Vec<Chunk>,
}

/// Represent borders of a `Range` request
impl Request {
    pub fn for_chunk(chunk: Chunk) -> Request {
        Request {
            first_chunk: chunk,
            other_chunks: vec![],
        }
    }
    pub fn last_index(&self) -> usize {
        self.other_chunks.last().unwrap_or(&self.first_chunk).index
    }
    pub fn bounds(&self) -> Range<usize> {
        self.first_chunk.chunk_left_ix..self.other_chunks.last().unwrap_or(&self.first_chunk).chunk_right_ix
    }
    pub fn chunks(&self) -> Chain<Once<&Chunk>, Iter<'_, Chunk>> {
        std::iter::once(&self.first_chunk).chain(self.other_chunks.iter())
    }
    pub fn is_chainable_with(&self, chunk: &Chunk) -> bool {
        self.last_index() == chunk.index - 1
    }
}

pub(crate) struct RequestsComposer {
    chunks: Vec<Chunk>,
}

impl RequestsComposer {
    pub fn for_chunks(chunks: Vec<Chunk>) -> RequestsComposer {
        RequestsComposer { chunks }
    }

    /// Used for chaining adjacent chunks into a single request
    pub fn requests(self) -> Vec<Request> {
        let mut merged_requests = vec![];
        let mut possible_merged_request = None;
        for chunk in self.chunks {
            possible_merged_request = match possible_merged_request {
                None => Some(Request::for_chunk(chunk.clone())),
                Some(mut merged_request) => {
                    if merged_request.is_chainable_with(&chunk) {
                        merged_request.other_chunks.push(chunk.clone());
                        Some(merged_request)
                    } else {
                        merged_requests.push(merged_request.clone());
                        Some(Request::for_chunk(chunk))
                    }
                }
            }
        }
        if let Some(merged_request) = possible_merged_request {
            merged_requests.push(merged_request);
        }
        merged_requests
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directories::chunk_generator::ChunkGenerator;

    #[test]
    fn test_request_merger() {
        let chunk_generator_1 = ChunkGenerator::new(2..10, 30, 3);
        let chunk_generator_2 = ChunkGenerator::new(20..30, 30, 3);
        let chunks_1 = chunk_generator_1.collect::<Vec<_>>();
        let chunks_2 = chunk_generator_2.collect::<Vec<_>>();
        let requests = compose_requests_for_chunks(chunks_1.into_iter().chain(chunks_2.into_iter()).collect::<Vec<_>>().into_iter());
        assert_eq!(
            &requests[..],
            &[
                Request {
                    first_chunk: Chunk {
                        index: 0,
                        chunk_left_ix: 0,
                        chunk_right_ix: 3,
                        inner_left_ix: 2,
                        inner_right_ix: 3,
                        target_ix: 0
                    },
                    other_chunks: vec![
                        Chunk {
                            index: 1,
                            chunk_left_ix: 3,
                            chunk_right_ix: 6,
                            inner_left_ix: 0,
                            inner_right_ix: 3,
                            target_ix: 1
                        },
                        Chunk {
                            index: 2,
                            chunk_left_ix: 6,
                            chunk_right_ix: 9,
                            inner_left_ix: 0,
                            inner_right_ix: 3,
                            target_ix: 4
                        },
                        Chunk {
                            index: 3,
                            chunk_left_ix: 9,
                            chunk_right_ix: 12,
                            inner_left_ix: 0,
                            inner_right_ix: 1,
                            target_ix: 7
                        }
                    ]
                },
                Request {
                    first_chunk: Chunk {
                        index: 6,
                        chunk_left_ix: 18,
                        chunk_right_ix: 21,
                        inner_left_ix: 2,
                        inner_right_ix: 3,
                        target_ix: 0
                    },
                    other_chunks: vec![
                        Chunk {
                            index: 7,
                            chunk_left_ix: 21,
                            chunk_right_ix: 24,
                            inner_left_ix: 0,
                            inner_right_ix: 3,
                            target_ix: 1
                        },
                        Chunk {
                            index: 8,
                            chunk_left_ix: 24,
                            chunk_right_ix: 27,
                            inner_left_ix: 0,
                            inner_right_ix: 3,
                            target_ix: 4
                        },
                        Chunk {
                            index: 9,
                            chunk_left_ix: 27,
                            chunk_right_ix: 30,
                            inner_left_ix: 0,
                            inner_right_ix: 3,
                            target_ix: 7
                        }
                    ]
                }
            ]
        );
    }
}
