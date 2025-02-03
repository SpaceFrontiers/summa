//! Data processing functions: simhash, sort, term-stats

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, BufWriter, Cursor, Write};
use std::path::PathBuf;

use anyhow::Result;

fn simhash(text: &str) -> u32 {
    let tokens: Vec<&str> = text.split_whitespace().collect();

    if tokens.is_empty() {
        return 0;
    }

    let mut bit_counts = [0i32; 32];

    for token in tokens {
        let hash = murmur3::murmur3_32(&mut Cursor::new(token.as_bytes()), 0).unwrap_or(0);

        for (i, count) in bit_counts.iter_mut().enumerate() {
            if (hash >> i) & 1 == 0 {
                *count += 1;
            } else {
                *count -= 1;
            }
        }
    }

    let mut fingerprint = 0u32;
    for (i, &count) in bit_counts.iter().enumerate() {
        if count > 0 {
            fingerprint |= 1 << i;
        }
    }

    fingerprint
}

pub fn run_simhash(field: &str, output_field: &str) -> Result<()> {
    use crossbeam_channel::{Receiver, Sender, bounded};
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    const BATCH_SIZE: usize = 10_000;
    const CHANNEL_CAPACITY: usize = 4;

    let field = field.to_string();
    let output_field = output_field.to_string();

    // Channels for pipeline: reader -> processor -> writer
    let (line_tx, line_rx): (Sender<Vec<String>>, Receiver<Vec<String>>) =
        bounded(CHANNEL_CAPACITY);
    let (result_tx, result_rx): (Sender<Vec<String>>, Receiver<Vec<String>>) =
        bounded(CHANNEL_CAPACITY);

    let count = std::sync::Arc::new(AtomicUsize::new(0));
    let errors = std::sync::Arc::new(AtomicUsize::new(0));

    let count_clone = count.clone();
    let errors_clone = errors.clone();

    // Reader thread
    let reader_handle = thread::spawn(move || -> Result<()> {
        let stdin = io::stdin();
        let reader = BufReader::new(stdin.lock());
        let mut batch = Vec::with_capacity(BATCH_SIZE);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            batch.push(line);

            if batch.len() >= BATCH_SIZE {
                if line_tx.send(std::mem::take(&mut batch)).is_err() {
                    break;
                }
                batch = Vec::with_capacity(BATCH_SIZE);
            }
        }

        if !batch.is_empty() {
            let _ = line_tx.send(batch);
        }

        Ok(())
    });

    // Processor thread (uses rayon for parallel simhash computation)
    let processor_handle = thread::spawn(move || {
        while let Ok(batch) = line_rx.recv() {
            let results: Vec<String> = batch
                .into_par_iter()
                .filter_map(|line| {
                    let mut json: serde_json::Value = match sonic_rs::from_str(&line) {
                        Ok(v) => v,
                        Err(_) => {
                            errors_clone.fetch_add(1, Ordering::Relaxed);
                            return None;
                        }
                    };

                    if let Some(obj) = json.as_object_mut() {
                        let text = obj.get(&field).and_then(|v| v.as_str()).unwrap_or("");
                        let hash = simhash(text);
                        obj.insert(output_field.clone(), serde_json::Value::from(hash));
                        count_clone.fetch_add(1, Ordering::Relaxed);
                        serde_json::to_string(&json).ok()
                    } else {
                        None
                    }
                })
                .collect();

            if result_tx.send(results).is_err() {
                break;
            }
        }
    });

    // Writer thread
    let writer_handle = thread::spawn(move || -> Result<()> {
        let stdout = io::stdout();
        let mut writer = BufWriter::new(stdout.lock());

        while let Ok(batch) = result_rx.recv() {
            for line in batch {
                writeln!(writer, "{}", line)?;
            }
        }

        writer.flush()?;
        Ok(())
    });

    reader_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Reader thread panicked"))??;
    processor_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Processor thread panicked"))?;
    writer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Writer thread panicked"))??;

    let final_count = count.load(Ordering::Relaxed);
    let final_errors = errors.load(Ordering::Relaxed);

    if final_errors > 0 {
        eprintln!(
            "Processed {} documents, {} errors",
            final_count, final_errors
        );
    } else {
        eprintln!("Processed {} documents", final_count);
    }

    Ok(())
}

fn compare_by_field(
    a: &serde_json::Value,
    b: &serde_json::Value,
    field: &str,
    numeric: bool,
    reverse: bool,
) -> std::cmp::Ordering {
    let val_a = a.get(field);
    let val_b = b.get(field);

    let cmp = if numeric {
        let num_a = val_a.and_then(|v| v.as_f64()).unwrap_or(0.0);
        let num_b = val_b.and_then(|v| v.as_f64()).unwrap_or(0.0);
        num_a
            .partial_cmp(&num_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    } else {
        let str_a = val_a.and_then(|v| v.as_str()).unwrap_or("");
        let str_b = val_b.and_then(|v| v.as_str()).unwrap_or("");
        str_a.cmp(str_b)
    };

    if reverse { cmp.reverse() } else { cmp }
}

fn write_chunk(
    mut chunk: Vec<serde_json::Value>,
    field: &str,
    numeric: bool,
    reverse: bool,
    temp_dir: &std::path::Path,
    chunk_num: usize,
) -> Result<PathBuf> {
    use rayon::prelude::*;

    // Use parallel sort
    chunk.par_sort_by(|a, b| compare_by_field(a, b, field, numeric, reverse));

    let chunk_path = temp_dir.join(format!("chunk_{:06}.jsonl", chunk_num));
    let file = std::fs::File::create(&chunk_path)?;
    let mut writer = BufWriter::new(file);

    for doc in chunk.iter() {
        serde_json::to_writer(&mut writer, doc)?;
        writeln!(writer)?;
    }
    writer.flush()?;

    Ok(chunk_path)
}

fn merge_chunks(
    chunk_paths: Vec<PathBuf>,
    field: &str,
    numeric: bool,
    reverse: bool,
) -> Result<()> {
    use std::collections::BinaryHeap;

    let mut readers: Vec<std::io::Lines<BufReader<std::fs::File>>> = Vec::new();
    for path in &chunk_paths {
        let file = std::fs::File::open(path)?;
        readers.push(BufReader::new(file).lines());
    }

    struct HeapItem {
        value: serde_json::Value,
        chunk_idx: usize,
        field: String,
        numeric: bool,
        reverse: bool,
    }

    impl PartialEq for HeapItem {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == std::cmp::Ordering::Equal
        }
    }
    impl Eq for HeapItem {}

    impl PartialOrd for HeapItem {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for HeapItem {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            compare_by_field(
                &other.value,
                &self.value,
                &self.field,
                self.numeric,
                self.reverse,
            )
        }
    }

    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    for (idx, reader) in readers.iter_mut().enumerate() {
        if let Some(Ok(line)) = reader.next()
            && let Ok(value) = sonic_rs::from_str(&line)
        {
            heap.push(HeapItem {
                value,
                chunk_idx: idx,
                field: field.to_string(),
                numeric,
                reverse,
            });
        }
    }

    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    let mut output_count = 0usize;

    while let Some(item) = heap.pop() {
        serde_json::to_writer(&mut writer, &item.value)?;
        writeln!(writer)?;
        output_count += 1;

        if let Some(Ok(line)) = readers[item.chunk_idx].next()
            && let Ok(value) = sonic_rs::from_str(&line)
        {
            heap.push(HeapItem {
                value,
                chunk_idx: item.chunk_idx,
                field: field.to_string(),
                numeric,
                reverse,
            });
        }
    }

    writer.flush()?;
    eprintln!(
        "Merged {} documents from {} chunks",
        output_count,
        chunk_paths.len()
    );

    for path in chunk_paths {
        let _ = std::fs::remove_file(path);
    }

    Ok(())
}

pub fn run_sort(
    field: &str,
    reverse: bool,
    numeric: bool,
    chunk_size: usize,
    temp_dir: Option<PathBuf>,
) -> Result<()> {
    use crossbeam_channel::{Sender, bounded};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    let stdin = io::stdin();
    let reader = BufReader::new(stdin.lock());

    let temp_dir = temp_dir.unwrap_or_else(|| std::env::temp_dir().join("hermes-sort"));
    std::fs::create_dir_all(&temp_dir)?;

    // Channel for sending chunks to be sorted/written in parallel
    #[allow(clippy::type_complexity)]
    let (chunk_tx, chunk_rx): (
        Sender<(Vec<serde_json::Value>, usize)>,
        crossbeam_channel::Receiver<(Vec<serde_json::Value>, usize)>,
    ) = bounded(4);

    // Channel for receiving completed chunk paths
    #[allow(clippy::type_complexity)]
    let (path_tx, path_rx): (
        Sender<(usize, PathBuf)>,
        crossbeam_channel::Receiver<(usize, PathBuf)>,
    ) = bounded(100);

    let field_clone = field.to_string();
    let temp_dir_clone = temp_dir.clone();
    let errors_count = Arc::new(AtomicUsize::new(0));

    // Spawn worker threads for parallel chunk sorting/writing
    let num_workers = rayon::current_num_threads().min(4);
    let mut worker_handles = Vec::new();

    for _ in 0..num_workers {
        let rx = chunk_rx.clone();
        let tx = path_tx.clone();
        let field = field_clone.clone();
        let temp_dir = temp_dir_clone.clone();

        let handle = thread::spawn(move || {
            while let Ok((chunk, chunk_num)) = rx.recv() {
                match write_chunk(chunk, &field, numeric, reverse, &temp_dir, chunk_num) {
                    Ok(path) => {
                        let _ = tx.send((chunk_num, path));
                    }
                    Err(e) => {
                        eprintln!("Error writing chunk {}: {}", chunk_num, e);
                    }
                }
            }
        });
        worker_handles.push(handle);
    }

    // Drop extra senders so workers can terminate
    drop(chunk_rx);
    drop(path_tx);

    let mut chunk: Vec<serde_json::Value> = Vec::with_capacity(chunk_size);
    let mut chunk_num = 0usize;
    let mut total_docs = 0usize;
    let mut errors = 0usize;

    eprintln!(
        "Reading and sorting chunks (chunk_size={}, workers={})...",
        chunk_size, num_workers
    );

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        match sonic_rs::from_str(&line) {
            Ok(v) => {
                chunk.push(v);
                total_docs += 1;

                if chunk.len() >= chunk_size {
                    eprintln!("  Sending chunk {} ({} docs)", chunk_num, chunk.len());
                    if chunk_tx
                        .send((std::mem::take(&mut chunk), chunk_num))
                        .is_err()
                    {
                        break;
                    }
                    chunk_num += 1;
                    chunk = Vec::with_capacity(chunk_size);
                }
            }
            Err(e) => {
                if errors < 5 {
                    eprintln!(
                        "Warning: Failed to parse JSON at line {}: {}",
                        line_num + 1,
                        e
                    );
                }
                errors += 1;
            }
        }
    }

    // Send final chunk
    if !chunk.is_empty() {
        eprintln!("  Sending chunk {} ({} docs)", chunk_num, chunk.len());
        let _ = chunk_tx.send((chunk, chunk_num));
    }

    // Close sender to signal workers to finish
    drop(chunk_tx);

    // Wait for all workers to complete
    for handle in worker_handles {
        let _ = handle.join();
    }

    // Collect all chunk paths and sort by chunk number to maintain order
    let mut chunk_paths: Vec<(usize, PathBuf)> = path_rx.iter().collect();
    chunk_paths.sort_by_key(|(num, _)| *num);
    let chunk_paths: Vec<PathBuf> = chunk_paths.into_iter().map(|(_, path)| path).collect();

    errors += errors_count.load(Ordering::Relaxed);

    eprintln!(
        "Read {} documents into {} chunks",
        total_docs,
        chunk_paths.len()
    );

    if chunk_paths.is_empty() {
        eprintln!("No documents to sort");
        return Ok(());
    }

    if chunk_paths.len() == 1 {
        let file = std::fs::File::open(&chunk_paths[0])?;
        let reader = BufReader::new(file);
        let stdout = io::stdout();
        let mut writer = BufWriter::new(stdout.lock());

        for line in reader.lines() {
            writeln!(writer, "{}", line?)?;
        }
        writer.flush()?;
        let _ = std::fs::remove_file(&chunk_paths[0]);
        eprintln!("Sorted {} documents (single chunk)", total_docs);
    } else {
        eprintln!("Merging {} chunks...", chunk_paths.len());
        merge_chunks(chunk_paths, field, numeric, reverse)?;
    }

    if errors > 0 {
        eprintln!("Skipped {} documents due to parse errors", errors);
    }

    let _ = std::fs::remove_dir(&temp_dir);

    Ok(())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode)]
pub struct TermStats {
    pub term: String,
    pub df: u32,
    pub total_tf: u64,
    pub max_tf: u32,
    pub idf: f32,
    pub upper_bound: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode)]
pub struct WandStats {
    pub total_docs: u64,
    pub total_tokens: u64,
    pub avg_doc_len: f32,
    pub bm25_k1: f32,
    pub bm25_b: f32,
    pub terms: Vec<TermStats>,
}

fn tokenize_simple(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

fn compute_idf(total_docs: u64, df: u32) -> f32 {
    let n = total_docs as f32;
    let df = df as f32;
    ((n - df + 0.5) / (df + 0.5)).ln()
}

fn compute_upper_bound(max_tf: u32, idf: f32, k1: f32, b: f32) -> f32 {
    let tf = max_tf as f32;
    let min_length_norm = 1.0 - b;
    let tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * min_length_norm);
    idf * tf_norm
}

pub fn run_term_stats(
    fields: &[String],
    format: &str,
    min_df: u32,
    bm25_k1: f32,
    bm25_b: f32,
) -> Result<()> {
    let stdin = io::stdin();
    let reader = BufReader::new(stdin.lock());

    let mut term_stats: HashMap<(String, String), (u32, u64, u32)> = HashMap::new();
    let mut total_docs = 0u64;
    let mut total_tokens = 0u64;
    let mut errors = 0usize;

    eprintln!("Computing term statistics for fields: {:?}", fields);

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let json: serde_json::Value = match sonic_rs::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                if errors < 5 {
                    eprintln!("Warning: Failed to parse JSON: {}", e);
                }
                errors += 1;
                continue;
            }
        };

        total_docs += 1;

        if let Some(obj) = json.as_object() {
            let mut doc_terms: HashMap<(String, String), u32> = HashMap::new();

            for field_name in fields {
                let text = obj.get(field_name).and_then(|v| v.as_str()).unwrap_or("");
                let tokens = tokenize_simple(text);
                total_tokens += tokens.len() as u64;

                for token in tokens {
                    let key = (field_name.clone(), token);
                    *doc_terms.entry(key).or_insert(0) += 1;
                }
            }

            for ((field, term), tf) in doc_terms {
                let entry = term_stats.entry((field, term)).or_insert((0, 0, 0));
                entry.0 += 1;
                entry.1 += tf as u64;
                entry.2 = entry.2.max(tf);
            }
        }

        if total_docs.is_multiple_of(100_000) {
            eprintln!(
                "  Processed {} documents, {} unique terms...",
                total_docs,
                term_stats.len()
            );
        }
    }

    eprintln!(
        "Processed {} documents, {} unique terms, {} total tokens",
        total_docs,
        term_stats.len(),
        total_tokens
    );

    let avg_doc_len = if total_docs > 0 {
        total_tokens as f32 / total_docs as f32
    } else {
        0.0
    };

    let mut terms: Vec<TermStats> = term_stats
        .into_iter()
        .filter(|(_, (df, _, _))| *df >= min_df)
        .map(|((field, term), (df, total_tf, max_tf))| {
            let idf = compute_idf(total_docs, df);
            let upper_bound = compute_upper_bound(max_tf, idf, bm25_k1, bm25_b);
            TermStats {
                term: format!("{}:{}", field, term),
                df,
                total_tf,
                max_tf,
                idf,
                upper_bound,
            }
        })
        .collect();

    terms.sort_by_key(|t| std::cmp::Reverse(t.df));

    let wand_stats = WandStats {
        total_docs,
        total_tokens,
        avg_doc_len,
        bm25_k1,
        bm25_b,
        terms,
    };

    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());

    match format {
        "json" => {
            serde_json::to_writer_pretty(&mut writer, &wand_stats)?;
            writeln!(writer)?;
        }
        "binary" => {
            let encoded = bincode::encode_to_vec(&wand_stats, bincode::config::standard())
                .map_err(|e| anyhow::anyhow!("Bincode error: {}", e))?;
            writer.write_all(&encoded)?;
        }
        _ => {
            anyhow::bail!("Unknown format: {}. Use 'json' or 'binary'", format);
        }
    }

    writer.flush()?;

    eprintln!(
        "Output {} terms (min_df={}), avg_doc_len={:.2}",
        wand_stats.terms.len(),
        min_df,
        avg_doc_len
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simhash_similar_strings() {
        let h1 = simhash("The quick brown fox jumps over the lazy dog");
        let h2 = simhash("The quick brown fox jumps over the lazy cat");
        let h3 = simhash("Something completely different here");

        let dist_similar = (h1 ^ h2).count_ones();
        let dist_different = (h1 ^ h3).count_ones();

        assert!(
            dist_similar < dist_different,
            "Similar strings should have smaller Hamming distance: {} vs {}",
            dist_similar,
            dist_different
        );
    }

    #[test]
    fn test_simhash_identical() {
        let h1 = simhash("Hello world");
        let h2 = simhash("Hello world");
        assert_eq!(h1, h2);
    }
}
