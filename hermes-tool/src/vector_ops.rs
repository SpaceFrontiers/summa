//! Vector index operations: train and retrain centroids

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use anyhow::{Context, Result};
use tracing::info;

use hermes_core::{FsDirectory, IndexConfig};

/// Train coarse centroids from a JSONL file containing vectors
pub fn train_centroids(
    input: PathBuf,
    field: String,
    output: PathBuf,
    clusters: Option<usize>,
    max_iters: usize,
    sample_size: Option<usize>,
    seed: u64,
) -> Result<()> {
    use hermes_core::structures::CoarseCentroids;

    info!("Loading vectors from {:?}, field '{}'", input, field);

    let file = File::open(&input).context("Failed to open input file")?;
    let reader = BufReader::new(file);

    let mut vectors: Vec<Vec<f32>> = Vec::new();
    let max_samples = sample_size.unwrap_or(usize::MAX);

    for line in reader.lines() {
        if vectors.len() >= max_samples {
            break;
        }

        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let json: serde_json::Value =
            serde_json::from_str(&line).context("Failed to parse JSON line")?;

        if let Some(vec_value) = json.get(&field)
            && let Some(arr) = vec_value.as_array()
        {
            let vec: Vec<f32> = arr
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            if !vec.is_empty() {
                vectors.push(vec);
            }
        }
    }

    if vectors.is_empty() {
        anyhow::bail!("No vectors found in input file");
    }

    let dim = vectors[0].len();
    info!("Loaded {} vectors of dimension {}", vectors.len(), dim);

    // Determine number of clusters
    let num_clusters = clusters.unwrap_or_else(|| {
        let sqrt = (vectors.len() as f64).sqrt() as usize;
        sqrt.clamp(16, 65536)
    });

    info!(
        "Training {} clusters with {} iterations (seed={})",
        num_clusters, max_iters, seed
    );

    let start = std::time::Instant::now();
    let coarse_config = hermes_core::structures::CoarseConfig::new(dim, num_clusters)
        .with_max_iters(max_iters)
        .with_seed(seed);
    let centroids = CoarseCentroids::train(&coarse_config, &vectors);
    let elapsed = start.elapsed();

    info!(
        "Training complete in {:.2}s, saving to {:?}",
        elapsed.as_secs_f64(),
        output
    );

    centroids
        .save(&output)
        .context("Failed to save centroids")?;

    eprintln!(
        "Trained {} clusters from {} vectors ({} dims) in {:.2}s",
        num_clusters,
        vectors.len(),
        dim,
        elapsed.as_secs_f64()
    );
    eprintln!("Saved to: {:?}", output);

    Ok(())
}

/// Retrain centroids from an existing index and rebuild vector indexes
pub async fn retrain_centroids(
    index_path: PathBuf,
    _field: String,
    clusters: Option<usize>,
    max_iters: usize,
    sample_size: Option<usize>,
    seed: u64,
) -> Result<()> {
    use hermes_core::structures::CoarseCentroids;

    info!("Opening index at {:?}", index_path);

    let dir = FsDirectory::new(index_path.clone());
    let config = IndexConfig::default();

    let index = hermes_core::Index::open(dir, config)
        .await
        .context("Failed to open index")?;

    // Collect vectors from all segments
    info!("Collecting vectors from index segments...");
    let vectors: Vec<Vec<f32>> = Vec::new();
    let _max_samples = sample_size.unwrap_or(1_000_000);

    // TODO: Implement vector extraction from segments
    // This requires iterating through documents and extracting dense vector fields
    // For now, we'll show a placeholder message

    if vectors.is_empty() {
        eprintln!("Note: Vector extraction from existing index not yet implemented.");
        eprintln!("Please use 'train-centroids' with a JSONL file containing vectors.");
        eprintln!();
        eprintln!("Example workflow:");
        eprintln!(
            "  1. Export vectors: hermes-tool export-vectors -i {} -f {} > vectors.jsonl",
            index_path.display(),
            _field
        );
        eprintln!(
            "  2. Train centroids: hermes-tool train-centroids -i vectors.jsonl -f {} -o centroids.bin",
            _field
        );
        eprintln!("  3. Rebuild index with new centroids");
        return Ok(());
    }

    let dim = vectors[0].len();
    let num_clusters = clusters.unwrap_or_else(|| {
        let sqrt = (vectors.len() as f64).sqrt() as usize;
        sqrt.clamp(16, 65536)
    });

    info!(
        "Training {} clusters from {} vectors",
        num_clusters,
        vectors.len()
    );

    let coarse_config = hermes_core::structures::CoarseConfig::new(dim, num_clusters)
        .with_max_iters(max_iters)
        .with_seed(seed);
    let centroids = CoarseCentroids::train(&coarse_config, &vectors);

    // Save centroids to index directory
    let centroids_path = index_path.join("coarse_centroids.bin");
    centroids.save(&centroids_path)?;

    info!("Saved centroids to {:?}", centroids_path);

    // TODO: Rebuild vector indexes with new centroids
    eprintln!(
        "Trained {} clusters, saved to {:?}",
        num_clusters, centroids_path
    );
    eprintln!("Note: Index rebuild with new centroids not yet implemented.");

    drop(index);
    Ok(())
}
