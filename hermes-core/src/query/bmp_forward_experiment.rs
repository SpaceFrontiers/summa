//! Manual experiment for the phase-two dispatch described in
//! docs/bmp-forward-search.md. This adds no retrieval policy or default.
use super::*;
use crate::directories::{FileHandle, OwnedBytes};
use std::hint::black_box;
use std::time::Instant;
