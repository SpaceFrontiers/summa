#!/usr/bin/env python3
"""
Generate benchmark data for dense and sparse vector indexes.

Downloads MS MARCO passages (or uses cached), generates:
- Dense embeddings via Triton server (jinaai/jina-embeddings-v3, Matryoshka-capable)
- Sparse vectors using naver/splade-v3 (local)

Usage:
    python generate_benchmark_data.py --num-docs 100000 --num-queries 1000

Output files:
    - dense_embeddings.bin: Dense vectors for documents (1024-dim, Jina v3)
    - dense_queries.bin: Dense vectors for queries
    - sparse_embeddings.bin: Sparse vectors for documents
    - sparse_queries.bin: Sparse vectors for queries
    - ground_truth_dense_full.bin: Ground truth for full-dim dense search
    - ground_truth_sparse.bin: Ground truth for sparse search

Requires:
    pip install tritonclient[grpc] transformers datasets numpy tqdm torch
"""

import argparse
import asyncio
import os
import struct
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ImportError:
    print("Please install: pip install torch transformers")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("Please install: pip install datasets")
    sys.exit(1)


# ============================================================================
# Triton Client for Remote Embeddings
# ============================================================================


class TritonEmbedClient:
    """Async client for Triton embedding model (jina-v3-ensemble) via gRPC.

    Jina v3 outputs 1024-dim embeddings (Matryoshka-capable).
    Requires TRITON_URL and TRITON_API_KEY environment variables.
    """

    TASK_IDS = {
        "retrieval.query": 0,
        "retrieval.passage": 1,
        "separation": 2,
        "classification": 3,
        "text-matching": 4,
    }

    TASK_INSTRUCTIONS = {
        "retrieval.query": "Represent the query for retrieving evidence documents: ",
        "retrieval.passage": "Represent the document for retrieval: ",
    }

    def __init__(
        self,
        triton_url: str = None,
        api_key: str = None,
        model_name: str = "jina-v3-ensemble",
        tokenizer_name: str = "jinaai/jina-embeddings-v3",
    ):
        try:
            import tritonclient.grpc.aio as grpcclient

            self._grpcclient = grpcclient
        except ImportError:
            print("Please install: pip install tritonclient[grpc] grpcio")
            sys.exit(1)

        self.triton_url = triton_url or os.environ.get("TRITON_URL")
        if not self.triton_url:
            print("ERROR: TRITON_URL environment variable not set")
            sys.exit(1)

        self.api_key = api_key or os.environ.get("TRITON_API_KEY")
        if not self.api_key:
            print("ERROR: TRITON_API_KEY environment variable not set")
            sys.exit(1)

        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        self._headers = {"x-api-key": self.api_key}
        self._client = None
        print(f"Triton client initialized: {self.triton_url}")

    def _get_channel_args(self):
        """Get gRPC channel args with retry policy."""
        import json

        retry_config = json.dumps(
            {
                "methodConfig": [
                    {
                        "name": [{}],  # Apply to all methods
                        "retryPolicy": {
                            "maxAttempts": 5,
                            "initialBackoff": "0.5s",
                            "maxBackoff": "30s",
                            "backoffMultiplier": 2,
                            "retryableStatusCodes": [
                                "UNAVAILABLE",
                                "UNKNOWN",
                                "RESOURCE_EXHAUSTED",
                            ],
                        },
                    }
                ]
            }
        )
        return [("grpc.service_config", retry_config)]

    async def _get_client(self):
        """Get or create the gRPC client."""
        if self._client is None:
            self._client = self._grpcclient.InferenceServerClient(
                url=self.triton_url,
                ssl=False,
                channel_args=self._get_channel_args(),
            )
        return self._client

    async def encode(
        self,
        texts: list[str],
        task: str = "retrieval.passage",
        max_batch_size: int = 32,
    ) -> np.ndarray:
        """Encode texts to embeddings using Triton server."""
        client = await self._get_client()
        grpcclient = self._grpcclient

        all_embeddings = []
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i : i + max_batch_size]

            # Prepend task instruction
            instruction = self.TASK_INSTRUCTIONS.get(task, "")
            prefixed_texts = [instruction + t for t in batch]

            encoded = self._tokenizer(
                prefixed_texts,
                padding=True,
                truncation=True,
                max_length=768,
                return_tensors="np",
            )

            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)
            task_id = np.array(
                [[self.TASK_IDS.get(task, 1)]] * len(batch), dtype=np.int64
            )

            inputs = [
                grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
                grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
                grpcclient.InferInput("task_id", task_id.shape, "INT64"),
            ]
            inputs[0].set_data_from_numpy(input_ids)
            inputs[1].set_data_from_numpy(attention_mask)
            inputs[2].set_data_from_numpy(task_id)

            outputs = [grpcclient.InferRequestedOutput("embedding")]
            result = await client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
                headers=self._headers,
            )
            all_embeddings.append(result.as_numpy("embedding"))

        return np.vstack(all_embeddings)


async def generate_dense_embeddings_triton(
    texts: list[str],
    batch_size: int = 32,
    is_query: bool = False,
    output_path: Path | None = None,
    checkpoint_interval: int = 1000,
) -> np.ndarray:
    """Generate dense embeddings using remote Triton server.

    Jina v3 on Triton outputs 1024-dim embeddings.
    Requires TRITON_URL and TRITON_API_KEY environment variables.
    Supports resuming from checkpoint if interrupted.
    """
    # Check for existing checkpoint
    start_idx = 0
    existing_embeddings = None
    if output_path:
        existing_embeddings, start_idx = load_dense_checkpoint(output_path)
        if start_idx > 0:
            print(
                f"Resuming from checkpoint: {start_idx}/{len(texts)} texts already processed"
            )

    if start_idx >= len(texts):
        print("All texts already processed, using cached embeddings")
        return existing_embeddings.astype(np.float32)

    client = TritonEmbedClient()
    task = "retrieval.query" if is_query else "retrieval.passage"

    remaining_texts = texts[start_idx:]
    print(
        f"Generating dense embeddings via Triton for {len(remaining_texts)} texts (task={task})..."
    )

    all_new_embeddings = []

    for i in tqdm(range(0, len(remaining_texts), batch_size), desc="Triton encoding"):
        batch = remaining_texts[i : i + batch_size]
        batch_embeddings = await client.encode(
            batch, task=task, max_batch_size=batch_size
        )
        all_new_embeddings.append(batch_embeddings)

        # Save checkpoint periodically
        processed = start_idx + i + len(batch)
        if output_path and (i + batch_size) % checkpoint_interval < batch_size:
            partial = np.vstack(all_new_embeddings)
            if existing_embeddings is not None:
                partial = np.vstack([existing_embeddings, partial])
            save_dense_checkpoint(partial, processed, output_path)

    # Combine all embeddings
    new_embeddings = np.vstack(all_new_embeddings)
    if existing_embeddings is not None:
        embeddings = np.vstack([existing_embeddings, new_embeddings])
    else:
        embeddings = new_embeddings

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    print(
        f"Generated {embeddings.shape[0]} vectors with {embeddings.shape[1]} dimensions"
    )
    return embeddings.astype(np.float32)


# ============================================================================
# Corpus Loading (MS MARCO Passage Ranking with qrels)
# ============================================================================


def load_msmarco_data(
    num_docs: int, num_queries: int, cache_dir: Path | None = None
) -> tuple[list[str], list[str], list[str], list[str], dict[int, list[int]]]:
    """Load MS MARCO passage ranking dataset with qrels.

    Strategy:
    1. Load ALL qrels (train + dev) to get more queries
    2. Select exactly num_queries queries that have qrels
    3. Collect all their relevant passages
    4. Fill remaining with random passages to reach num_docs total

    Returns:
        passages: list of passage texts (num_docs total, includes relevant ones)
        passage_ids: list of passage IDs
        queries: list of query texts (exactly num_queries)
        query_ids: list of query IDs
        qrels: dict mapping query_idx -> list of relevant passage_idxs
    """
    print("Loading MS MARCO passage ranking dataset...")
    print(f"  Target: {num_queries} queries, {num_docs} passages")

    # Load ALL qrels (train has more data)
    print("Loading qrels (train + dev)...")
    qrels_dataset = load_dataset(
        "mteb/msmarco-v2",
        "default",
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
    )

    # Build qrels dict from all splits
    print("Building qrels mapping...")
    qrels_raw: dict[str, list[str]] = {}
    for split_name in ["train", "dev", "dev2"]:
        if split_name in qrels_dataset:
            for item in tqdm(
                qrels_dataset[split_name], desc=f"Processing {split_name}"
            ):
                qid = item["query-id"]
                pid = item["corpus-id"]
                if qid not in qrels_raw:
                    qrels_raw[qid] = []
                qrels_raw[qid].append(pid)

    print(f"  Found {len(qrels_raw)} queries with relevance judgments")

    # Load queries
    print("Loading queries...")
    queries_dataset = load_dataset(
        "mteb/msmarco-v2",
        "queries",
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
    )["queries"]

    # Build query lookup (only for queries with qrels)
    query_texts: dict[str, str] = {}
    for item in tqdm(queries_dataset, desc="Indexing queries"):
        qid = item["_id"]
        if qid in qrels_raw:
            query_texts[qid] = item["text"]

    # Select exactly num_queries queries
    selected_query_ids = [qid for qid in qrels_raw if qid in query_texts][:num_queries]
    print(f"Selected {len(selected_query_ids)} queries")

    # Collect all passage IDs needed for these queries
    needed_passage_ids: set[str] = set()
    for qid in selected_query_ids:
        for pid in qrels_raw[qid]:
            needed_passage_ids.add(pid)

    print(f"Need {len(needed_passage_ids)} relevant passages for selected queries")

    # Load corpus
    print("Loading corpus...")
    corpus = load_dataset(
        "mteb/msmarco-v2",
        "corpus",
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
    )["corpus"]

    # Collect passages: first the needed ones, then random ones
    print("Extracting passages...")
    needed_passages: dict[str, str] = {}
    random_passages: list[tuple[str, str]] = []

    import random

    random.seed(42)

    # Calculate how many random passages we need
    num_random_needed = num_docs - len(needed_passage_ids)
    random_sample_rate = (num_random_needed * 2) / 138_000_000  # 2x oversample

    for item in tqdm(corpus, desc="Loading passages"):
        pid = item["_id"]
        text = item.get("text", "") or item.get("title", "")
        if not text:
            continue

        if pid in needed_passage_ids:
            needed_passages[pid] = text
        elif random.random() < random_sample_rate:
            random_passages.append((pid, text))

        # Stop early if we have everything
        if (
            len(needed_passages) >= len(needed_passage_ids)
            and len(random_passages) >= num_random_needed
        ):
            break

    print(f"  Collected {len(needed_passages)} relevant passages")
    print(f"  Collected {len(random_passages)} random passages")

    # Build final passage list: relevant passages first, then random
    passages = []
    passage_ids = []
    passage_id_to_idx: dict[str, int] = {}

    # Add relevant passages first
    for pid, text in needed_passages.items():
        passage_id_to_idx[pid] = len(passages)
        passage_ids.append(pid)
        passages.append(text)

    # Fill up to num_docs with random passages
    for pid, text in random_passages:
        if len(passages) >= num_docs:
            break
        if pid not in passage_id_to_idx:  # Avoid duplicates
            passage_id_to_idx[pid] = len(passages)
            passage_ids.append(pid)
            passages.append(text)

    print(
        f"Total passages: {len(passages)} ({len(needed_passages)} relevant + {len(passages) - len(needed_passages)} random)"
    )

    # Build final queries and qrels
    queries = []
    query_ids_out = []
    valid_qrels: dict[int, list[int]] = {}

    for qid in selected_query_ids:
        if qid not in query_texts:
            continue

        relevant_idxs = []
        for pid in qrels_raw[qid]:
            if pid in passage_id_to_idx:
                relevant_idxs.append(passage_id_to_idx[pid])

        if relevant_idxs:
            query_idx = len(queries)
            queries.append(query_texts[qid])
            query_ids_out.append(qid)
            valid_qrels[query_idx] = relevant_idxs

    print(f"Final: {len(queries)} queries with valid qrels")
    avg_relevant = (
        sum(len(v) for v in valid_qrels.values()) / len(valid_qrels)
        if valid_qrels
        else 0
    )
    print(f"  Average relevant passages per query: {avg_relevant:.1f}")

    return passages, passage_ids, queries, query_ids_out, valid_qrels


def load_msmarco_passages(num_docs: int, cache_dir: Path | None = None) -> list[str]:
    """Load MS MARCO passages dataset (legacy interface)."""
    print(f"Loading MS MARCO passages (requesting {num_docs} documents)...")

    # Load the passage corpus
    dataset = load_dataset(
        "microsoft/ms_marco",
        "v1.1",
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
    )

    # Extract passages - MS MARCO has nested structure
    passages = []
    seen = set()

    for item in tqdm(dataset, desc="Extracting passages"):
        if len(passages) >= num_docs:
            break
        # Each item has 'passages' which is a dict with 'passage_text' list
        if "passages" in item and "passage_text" in item["passages"]:
            for text in item["passages"]["passage_text"]:
                if text and text not in seen:
                    seen.add(text)
                    passages.append(text)
                    if len(passages) >= num_docs:
                        break

    print(f"Loaded {len(passages)} unique passages")
    return passages


def load_msmarco_queries(num_queries: int, cache_dir: Path | None = None) -> list[str]:
    """Load MS MARCO queries (legacy interface)."""
    print(f"Loading MS MARCO queries (requesting {num_queries})...")

    dataset = load_dataset(
        "microsoft/ms_marco",
        "v1.1",
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
    )

    queries = []
    seen = set()

    for item in tqdm(dataset, desc="Extracting queries"):
        if len(queries) >= num_queries:
            break
        query = item.get("query", "")
        if query and query not in seen:
            seen.add(query)
            queries.append(query)

    print(f"Loaded {len(queries)} unique queries")
    return queries


def truncate_embeddings(
    embeddings: np.ndarray, target_dim: int, silent: bool = False
) -> np.ndarray:
    """Apply Matryoshka truncation to embeddings.

    Jina v3 supports dimensions: 32, 64, 128, 256, 512, 768, 1024
    """
    if embeddings.shape[1] <= target_dim:
        return embeddings

    if not silent:
        print(
            f"Truncating from {embeddings.shape[1]} to {target_dim} dimensions (Matryoshka)"
        )
    truncated = embeddings[:, :target_dim]
    # Re-normalize after truncation
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    truncated = truncated / np.maximum(norms, 1e-10)
    return truncated.astype(np.float32)


# ============================================================================
# Sparse Embeddings (SPLADE-v3)
# ============================================================================


def get_device() -> str:
    """Get the best available device for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SpladeEncoder:
    """SPLADE-v3 sparse encoder with GPU/MPS optimization."""

    def __init__(self, model_name: str = "naver/splade-v3"):
        print(f"Loading SPLADE model: {model_name}")
        self.device = get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
            if self.device in ("cuda", "mps")
            else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"SPLADE model loaded on {self.device} (fp16: {self.device != 'cpu'})")

    def encode(
        self, texts: list[str], batch_size: int = 32
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Encode texts to sparse vectors.

        Returns list of (indices, values) tuples.
        """
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="SPLADE encoding"):
            batch = texts[i : i + batch_size]
            batch_results = self._encode_batch(batch)
            results.extend(batch_results)

        return results

    def _encode_batch(self, texts: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Encode a batch of texts."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # SPLADE uses log(1 + ReLU(x)) aggregated over tokens
            logits = outputs.logits
            # Max pooling over sequence length with attention mask
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            logits = logits * attention_mask
            # Apply SPLADE activation: log(1 + ReLU(x))
            splade_rep = torch.log1p(torch.relu(logits))
            # Max pool over sequence
            splade_rep = torch.max(splade_rep, dim=1).values

        results = []
        for rep in splade_rep:
            # Get non-zero indices and values
            rep_cpu = rep.cpu().numpy()
            nonzero_mask = rep_cpu > 0.0
            indices = np.where(nonzero_mask)[0].astype(np.uint32)
            values = rep_cpu[nonzero_mask].astype(np.float32)

            # Sort by index for consistent ordering
            sort_idx = np.argsort(indices)
            indices = indices[sort_idx]
            values = values[sort_idx]

            results.append((indices, values))

        return results


def generate_sparse_embeddings(
    texts: list[str],
    model_name: str = "naver/splade-v3",
    batch_size: int = 32,
    output_path: Path | None = None,
    checkpoint_interval: int = 1000,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate sparse embeddings using SPLADE-v3.

    Supports resuming from checkpoint if interrupted.
    """
    # Check for existing checkpoint
    start_idx = 0
    existing_vectors: list[tuple[np.ndarray, np.ndarray]] = []
    if output_path:
        loaded, start_idx = load_sparse_checkpoint(output_path)
        if loaded:
            existing_vectors = loaded
            print(
                f"Resuming from checkpoint: {start_idx}/{len(texts)} texts already processed"
            )

    if start_idx >= len(texts):
        print("All texts already processed, using cached embeddings")
        return existing_vectors

    encoder = SpladeEncoder(model_name)
    remaining_texts = texts[start_idx:]

    results = list(existing_vectors)

    for i in tqdm(range(0, len(remaining_texts), batch_size), desc="SPLADE encoding"):
        batch = remaining_texts[i : i + batch_size]
        batch_results = encoder._encode_batch(batch)
        results.extend(batch_results)

        # Save checkpoint periodically
        processed = start_idx + i + len(batch)
        if output_path and (i + batch_size) % checkpoint_interval < batch_size:
            save_sparse_checkpoint(results, processed, output_path)

    return results


# ============================================================================
# Ground Truth Computation
# ============================================================================


def compute_dense_ground_truth(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    k: int = 100,
) -> np.ndarray:
    """Compute ground truth for dense vectors using brute force."""
    print(f"Computing dense ground truth (k={k})...")

    num_queries = query_embeddings.shape[0]
    ground_truth = np.zeros((num_queries, k), dtype=np.uint32)

    for i in tqdm(range(num_queries), desc="Dense ground truth"):
        # Compute dot product (vectors are normalized, so this is cosine similarity)
        scores = np.dot(doc_embeddings, query_embeddings[i])
        # Get top-k indices
        top_k = np.argsort(-scores)[:k]
        ground_truth[i] = top_k

    return ground_truth


def compute_sparse_ground_truth(
    query_vectors: list[tuple[np.ndarray, np.ndarray]],
    doc_vectors: list[tuple[np.ndarray, np.ndarray]],
    k: int = 100,
) -> np.ndarray:
    """Compute ground truth for sparse vectors using brute force dot product."""
    print(f"Computing sparse ground truth (k={k})...")

    num_queries = len(query_vectors)
    num_docs = len(doc_vectors)
    ground_truth = np.zeros((num_queries, k), dtype=np.uint32)

    for qi in tqdm(range(num_queries), desc="Sparse ground truth"):
        q_indices, q_values = query_vectors[qi]
        scores = np.zeros(num_docs, dtype=np.float32)

        for di in range(num_docs):
            d_indices, d_values = doc_vectors[di]
            # Compute sparse dot product
            score = sparse_dot(q_indices, q_values, d_indices, d_values)
            scores[di] = score

        # Get top-k indices
        top_k = np.argsort(-scores)[:k]
        ground_truth[qi] = top_k

    return ground_truth


def sparse_dot(
    indices1: np.ndarray,
    values1: np.ndarray,
    indices2: np.ndarray,
    values2: np.ndarray,
) -> float:
    """Compute dot product of two sparse vectors."""
    i, j = 0, 0
    result = 0.0

    while i < len(indices1) and j < len(indices2):
        if indices1[i] == indices2[j]:
            result += values1[i] * values2[j]
            i += 1
            j += 1
        elif indices1[i] < indices2[j]:
            i += 1
        else:
            j += 1

    return result


# ============================================================================
# File I/O with Checkpoint Support
# ============================================================================


def get_checkpoint_path(path: Path) -> Path:
    """Get checkpoint file path for a given output file."""
    return path.with_suffix(path.suffix + ".checkpoint.npz")


def save_dense_checkpoint(embeddings: np.ndarray, processed: int, path: Path):
    """Save dense embeddings checkpoint."""
    checkpoint_path = get_checkpoint_path(path)
    np.savez(checkpoint_path, embeddings=embeddings, processed=processed)


def load_dense_checkpoint(path: Path) -> tuple[np.ndarray | None, int]:
    """Load dense embeddings checkpoint if exists."""
    checkpoint_path = get_checkpoint_path(path)
    if checkpoint_path.exists():
        data = np.load(checkpoint_path)
        return data["embeddings"], int(data["processed"])
    return None, 0


def save_sparse_checkpoint(
    vectors: list[tuple[np.ndarray, np.ndarray]], processed: int, path: Path
):
    """Save sparse embeddings checkpoint."""
    checkpoint_path = get_checkpoint_path(path)
    # Convert to arrays for saving
    indices_list = [v[0] for v in vectors]
    values_list = [v[1] for v in vectors]
    np.savez(
        checkpoint_path,
        indices=np.array(indices_list, dtype=object),
        values=np.array(values_list, dtype=object),
        processed=processed,
    )


def load_sparse_checkpoint(
    path: Path,
) -> tuple[list[tuple[np.ndarray, np.ndarray]] | None, int]:
    """Load sparse embeddings checkpoint if exists."""
    checkpoint_path = get_checkpoint_path(path)
    if checkpoint_path.exists():
        data = np.load(checkpoint_path, allow_pickle=True)
        indices_list = data["indices"]
        values_list = data["values"]
        vectors = [
            (idx, val) for idx, val in zip(indices_list, values_list, strict=True)
        ]
        return vectors, int(data["processed"])
    return None, 0


def cleanup_checkpoint(path: Path):
    """Remove checkpoint file after successful completion."""
    checkpoint_path = get_checkpoint_path(path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  Cleaned up checkpoint: {checkpoint_path}")


def save_dense_embeddings(embeddings: np.ndarray, path: Path):
    """Save dense embeddings in binary format.

    Format: num_vectors (u32), dim (u32), vectors (f32 * num_vectors * dim)
    """
    num_vectors, dim = embeddings.shape

    with open(path, "wb") as f:
        f.write(struct.pack("<II", num_vectors, dim))
        f.write(embeddings.tobytes())

    print(f"Saved {num_vectors} dense vectors (dim={dim}) to {path}")
    print(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")
    cleanup_checkpoint(path)


def load_dense_embeddings(path: Path) -> np.ndarray:
    """Load dense embeddings from binary format."""
    with open(path, "rb") as f:
        num_vectors, dim = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.float32)
        embeddings = data.reshape(num_vectors, dim)
    print(f"  Loaded {num_vectors} vectors (dim={dim})")
    return embeddings


def save_sparse_embeddings(vectors: list[tuple[np.ndarray, np.ndarray]], path: Path):
    """Save sparse embeddings in binary format.

    Format:
    - num_vectors: u32
    - For each vector:
        - num_nonzero: u32
        - indices: [u32; num_nonzero]
        - values: [f32; num_nonzero]
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(vectors)))

        total_nnz = 0
        for indices, values in vectors:
            num_nonzero = len(indices)
            total_nnz += num_nonzero
            f.write(struct.pack("<I", num_nonzero))
            f.write(indices.astype(np.uint32).tobytes())
            f.write(values.astype(np.float32).tobytes())

    avg_nnz = total_nnz / len(vectors) if vectors else 0
    print(f"Saved {len(vectors)} sparse vectors to {path}")
    print(f"  Avg non-zeros: {avg_nnz:.1f}")
    print(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")
    cleanup_checkpoint(path)


def load_sparse_embeddings(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load sparse embeddings from binary format."""
    vectors = []
    with open(path, "rb") as f:
        num_vectors = struct.unpack("<I", f.read(4))[0]
        for _ in range(num_vectors):
            num_nonzero = struct.unpack("<I", f.read(4))[0]
            indices = np.frombuffer(f.read(num_nonzero * 4), dtype=np.uint32).copy()
            values = np.frombuffer(f.read(num_nonzero * 4), dtype=np.float32).copy()
            vectors.append((indices, values))
    print(f"  Loaded {num_vectors} sparse vectors")
    return vectors


def save_ground_truth(ground_truth: np.ndarray, path: Path):
    """Save ground truth in binary format.

    Format: num_queries (u32), k (u32), indices (u32 * num_queries * k)
    """
    num_queries, k = ground_truth.shape

    with open(path, "wb") as f:
        f.write(struct.pack("<II", num_queries, k))
        f.write(ground_truth.astype(np.uint32).tobytes())

    print(f"Saved ground truth ({num_queries} queries, k={k}) to {path}")


def save_qrels(qrels: dict[int, list[int]], path: Path):
    """Save qrels (relevance judgments) in binary format.

    Format:
    - num_queries: u32
    - For each query:
        - query_idx: u32
        - num_relevant: u32
        - relevant_passage_idxs: [u32; num_relevant]
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(qrels)))
        for query_idx, relevant_idxs in qrels.items():
            f.write(struct.pack("<II", query_idx, len(relevant_idxs)))
            f.write(np.array(relevant_idxs, dtype=np.uint32).tobytes())

    total_relevant = sum(len(v) for v in qrels.values())
    print(
        f"Saved qrels ({len(qrels)} queries, {total_relevant} total relevant) to {path}"
    )


def load_qrels(path: Path) -> dict[int, list[int]]:
    """Load qrels from binary format."""
    qrels = {}
    with open(path, "rb") as f:
        num_queries = struct.unpack("<I", f.read(4))[0]
        for _ in range(num_queries):
            query_idx, num_relevant = struct.unpack("<II", f.read(8))
            relevant_idxs = list(
                np.frombuffer(f.read(num_relevant * 4), dtype=np.uint32)
            )
            qrels[query_idx] = relevant_idxs
    print(f"  Loaded qrels for {len(qrels)} queries")
    return qrels


# ============================================================================
# Main
# ============================================================================


async def async_main(args):
    """Async main function."""
    # Pre-download tokenizers/models
    print("\n" + "=" * 60)
    print("Initializing")
    print("=" * 60)

    if not args.skip_dense:
        triton_url = os.environ.get("TRITON_URL")
        if not triton_url:
            print("ERROR: Set TRITON_URL environment variable")
            sys.exit(1)
        if not os.environ.get("TRITON_API_KEY"):
            print("ERROR: Set TRITON_API_KEY environment variable")
            sys.exit(1)
        print(f"Dense embeddings via Triton: {triton_url}")
        AutoTokenizer.from_pretrained(
            "jinaai/jina-embeddings-v3", trust_remote_code=True
        )
        print("  Tokenizer ready")

    if not args.skip_sparse:
        print(f"Downloading sparse model: {args.sparse_model}")
        AutoTokenizer.from_pretrained(args.sparse_model)
        AutoModelForMaskedLM.from_pretrained(args.sparse_model)
        print("  Sparse model ready")

    print("Ready!\n")

    # Load corpus
    qrels = None
    if args.use_beir:
        passages, _, queries, _, qrels = load_msmarco_data(
            args.num_docs, args.num_queries, args.cache_dir
        )
        # Save qrels for benchmark use
        qrels_path = args.output_dir / "qrels.bin"
        save_qrels(qrels, qrels_path)
    else:
        passages = load_msmarco_passages(args.num_docs, args.cache_dir)
        queries = load_msmarco_queries(args.num_queries, args.cache_dir)

    # Save raw texts for BM25 benchmarking
    texts_path = args.output_dir / "passages.txt"
    if not texts_path.exists():
        print(f"Saving {len(passages)} passages to {texts_path}")
        with open(texts_path, "w", encoding="utf-8") as f:
            for p in passages:
                # Replace newlines with spaces to keep one passage per line
                f.write(p.replace("\n", " ").replace("\r", "") + "\n")

    queries_text_path = args.output_dir / "queries.txt"
    if not queries_text_path.exists():
        print(f"Saving {len(queries)} queries to {queries_text_path}")
        with open(queries_text_path, "w", encoding="utf-8") as f:
            for q in queries:
                f.write(q.replace("\n", " ").replace("\r", "") + "\n")

    # Ensure we have enough data
    if len(passages) < args.num_docs:
        print(f"Warning: Only got {len(passages)} passages (requested {args.num_docs})")
    if len(queries) < args.num_queries:
        print(
            f"Warning: Only got {len(queries)} queries (requested {args.num_queries})"
        )

    # Generate dense embeddings
    if not args.skip_dense:
        print("\n" + "=" * 60)
        print("Generating Dense Embeddings")
        print("=" * 60)

        doc_output = args.output_dir / "dense_embeddings.bin"
        query_output = args.output_dir / "dense_queries.bin"

        # Check if embeddings already exist
        if doc_output.exists() and query_output.exists():
            print(f"Loading existing dense embeddings from {doc_output}")
            doc_embeddings = load_dense_embeddings(doc_output)
            print(f"Loading existing dense queries from {query_output}")
            query_embeddings = load_dense_embeddings(query_output)
        else:
            # Use Triton server for embeddings
            doc_embeddings = await generate_dense_embeddings_triton(
                passages,
                batch_size=args.batch_size,
                is_query=False,
                output_path=doc_output,
            )
            save_dense_embeddings(doc_embeddings, doc_output)

            query_embeddings = await generate_dense_embeddings_triton(
                queries,
                batch_size=args.batch_size,
                is_query=True,
                output_path=query_output,
            )
            save_dense_embeddings(query_embeddings, query_output)

        # Compute ground truth for full-dimensional embeddings (skip if exists)
        gt_full_path = args.output_dir / "ground_truth_dense_full.bin"
        if gt_full_path.exists():
            print(f"Full-dim ground truth already exists: {gt_full_path}")
        else:
            print(
                f"Computing ground truth for full {doc_embeddings.shape[1]}-dim embeddings..."
            )
            ground_truth_full = compute_dense_ground_truth(
                query_embeddings, doc_embeddings, args.k
            )
            save_ground_truth(ground_truth_full, gt_full_path)

        # Compute ground truth for truncated embeddings (skip if exists)
        gt_truncated_path = args.output_dir / f"ground_truth_dense_{args.dense_dim}.bin"
        if gt_truncated_path.exists():
            print(f"Truncated ground truth already exists: {gt_truncated_path}")
        else:
            print(
                f"Computing ground truth for truncated {args.dense_dim}-dim embeddings..."
            )
            doc_truncated = truncate_embeddings(
                doc_embeddings, args.dense_dim, silent=True
            )
            query_truncated = truncate_embeddings(
                query_embeddings, args.dense_dim, silent=True
            )
            ground_truth_truncated = compute_dense_ground_truth(
                query_truncated, doc_truncated, args.k
            )
            save_ground_truth(ground_truth_truncated, gt_truncated_path)

    # Generate sparse embeddings
    if not args.skip_sparse:
        print("\n" + "=" * 60)
        print("Generating Sparse Embeddings (SPLADE)")
        print("=" * 60)

        doc_sparse_output = args.output_dir / "sparse_embeddings.bin"
        query_sparse_output = args.output_dir / "sparse_queries.bin"

        # Check if sparse embeddings already exist
        if doc_sparse_output.exists() and query_sparse_output.exists():
            print(f"Loading existing sparse embeddings from {doc_sparse_output}")
            doc_sparse = load_sparse_embeddings(doc_sparse_output)
            print(f"Loading existing sparse queries from {query_sparse_output}")
            query_sparse = load_sparse_embeddings(query_sparse_output)
        else:
            doc_sparse = generate_sparse_embeddings(
                passages,
                model_name=args.sparse_model,
                batch_size=args.batch_size,
                output_path=doc_sparse_output,
            )
            save_sparse_embeddings(doc_sparse, doc_sparse_output)

            query_sparse = generate_sparse_embeddings(
                queries,
                model_name=args.sparse_model,
                batch_size=args.batch_size,
                output_path=query_sparse_output,
            )
            save_sparse_embeddings(query_sparse, query_sparse_output)

        # Compute ground truth (skip if already exists)
        sparse_gt_path = args.output_dir / "ground_truth_sparse.bin"
        if sparse_gt_path.exists():
            print(f"Sparse ground truth already exists: {sparse_gt_path}")
        else:
            ground_truth = compute_sparse_ground_truth(query_sparse, doc_sparse, args.k)
            save_ground_truth(ground_truth, sparse_gt_path)

    print("\n" + "=" * 60)
    print("Benchmark data generation complete!")
    print("=" * 60)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nTo run benchmarks:")
    print(f"  BENCHMARK_DATA={args.output_dir} cargo bench --bench large_benchmark")


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark data for vector indexes"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("benchmark_data"),
        help="Output directory",
    )
    parser.add_argument(
        "--num-docs",
        "-n",
        type=int,
        default=100000,
        help="Number of documents to index",
    )
    parser.add_argument(
        "--num-queries", "-q", type=int, default=10000, help="Number of queries"
    )
    parser.add_argument(
        "--dense-dim",
        "-d",
        type=int,
        default=128,
        help="Dense embedding dimension for ground truth (Matryoshka truncation)",
    )
    parser.add_argument(
        "--k", type=int, default=100, help="Number of ground truth neighbors"
    )
    parser.add_argument(
        "--sparse-model",
        type=str,
        default="naver/splade-v3",
        help="Sparse embedding model (local)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size for Triton encoding",
    )
    parser.add_argument(
        "--skip-dense", action="store_true", help="Skip dense embedding generation"
    )
    parser.add_argument(
        "--skip-sparse", action="store_true", help="Skip sparse embedding generation"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="Cache directory for datasets"
    )
    parser.add_argument(
        "--use-beir",
        action="store_true",
        help="Use BeIR/msmarco dataset with official qrels for IR evaluation",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
