#!/usr/bin/env python3
"""
Generate embeddings using Qwen3-Embedding model with Matryoshka truncation.

Usage:
    python generate_embeddings.py --output embeddings.bin --num-vectors 10000 --dim 128

Requires:
    pip install sentence-transformers torch
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print(
        "Please install sentence-transformers: pip install sentence-transformers torch"
    )
    sys.exit(1)


# Sample texts for generating embeddings (mix of topics)
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language",
    "The stock market experienced significant volatility today",
    "Climate change is affecting global weather patterns",
    "Quantum computing promises exponential speedups",
    "The human genome contains approximately 3 billion base pairs",
    "Renewable energy sources include solar and wind power",
    "Deep learning models require large amounts of training data",
    "The Internet of Things connects billions of devices",
    "Blockchain technology enables decentralized transactions",
    "Natural language processing understands human language",
    "Autonomous vehicles use sensors and AI for navigation",
    "Cybersecurity protects systems from digital attacks",
    "Virtual reality creates immersive digital experiences",
    "Edge computing processes data closer to the source",
    "5G networks offer faster speeds and lower latency",
    "Biotechnology advances medical treatments and diagnostics",
    "Robotics automates manufacturing and logistics",
    "Data science extracts insights from large datasets",
]


def generate_texts(num_texts: int) -> list[str]:
    """Generate diverse texts by combining and varying sample texts."""

    texts = []
    for i in range(num_texts):
        # Pick a base text and add variation
        base = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]

        # Add some variation
        variations = [
            f"{base}",
            f"In summary, {base.lower()}",
            f"It is known that {base.lower()}",
            f"Research shows that {base.lower()}",
            f"According to experts, {base.lower()}",
            f"Studies indicate that {base.lower()}",
            f"The evidence suggests that {base.lower()}",
            f"Many believe that {base.lower()}",
        ]

        text = variations[i % len(variations)]

        # Add unique suffix for diversity
        text += f" (sample {i})"
        texts.append(text)

    return texts


def generate_embeddings(
    model_name: str,
    texts: list[str],
    target_dim: int,
    batch_size: int = 32,
) -> np.ndarray:
    """Generate embeddings using the specified model."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    print(f"Generating embeddings for {len(texts)} texts...")

    # Generate embeddings in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True)
        all_embeddings.append(embeddings)

        if (i + batch_size) % 1000 == 0 or i + batch_size >= len(texts):
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.vstack(all_embeddings)

    # Matryoshka truncation - just take first target_dim dimensions
    if embeddings.shape[1] > target_dim:
        print(
            f"Truncating from {embeddings.shape[1]} to {target_dim} dimensions (Matryoshka)"
        )
        embeddings = embeddings[:, :target_dim]

        # Re-normalize after truncation
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)

    return embeddings.astype(np.float32)


def save_embeddings(embeddings: np.ndarray, output_path: Path):
    """
    Save embeddings in binary format.

    Format:
    - num_vectors: u32
    - dim: u32
    - vectors: [f32; num_vectors * dim]
    """
    num_vectors, dim = embeddings.shape

    with open(output_path, "wb") as f:
        f.write(struct.pack("<II", num_vectors, dim))
        f.write(embeddings.tobytes())

    print(f"Saved {num_vectors} vectors of dim {dim} to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings using Qwen3-Embedding"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("embeddings.bin"),
        help="Output file path",
    )
    parser.add_argument(
        "--num-vectors",
        "-n",
        type=int,
        default=10000,
        help="Number of vectors to generate",
    )
    parser.add_argument(
        "--dim",
        "-d",
        type=int,
        default=128,
        help="Target dimension (Matryoshka truncation)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        help="Model name (default: Alibaba-NLP/gte-Qwen2-1.5B-instruct)",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32, help="Batch size for encoding"
    )

    args = parser.parse_args()

    # Generate texts
    texts = generate_texts(args.num_vectors)

    # Generate embeddings
    embeddings = generate_embeddings(
        model_name=args.model,
        texts=texts,
        target_dim=args.dim,
        batch_size=args.batch_size,
    )

    # Save to file
    save_embeddings(embeddings, args.output)

    # Also save as queries (subset)
    num_queries = min(1000, args.num_vectors // 10)
    query_path = args.output.with_suffix(".queries.bin")
    save_embeddings(embeddings[:num_queries], query_path)


if __name__ == "__main__":
    main()
