#!/usr/bin/env python3
"""Minimal test for Triton client."""

import asyncio
import os
import sys

import numpy as np
import tritonclient.grpc.aio as grpcclient
from transformers import AutoTokenizer

# Check env vars
triton_url = os.environ.get("TRITON_URL")
api_key = os.environ.get("TRITON_API_KEY")

if not triton_url or not api_key:
    print("Set TRITON_URL and TRITON_API_KEY")
    sys.exit(1)

print(f"Testing Triton at {triton_url}")

# Setup
tokenizer = AutoTokenizer.from_pretrained(
    "jinaai/jina-embeddings-v3", trust_remote_code=True
)
headers = {"x-api-key": api_key}

TASK_IDS = {
    "retrieval.query": 0,
    "retrieval.passage": 1,
}


async def test_encode():
    client = grpcclient.InferenceServerClient(url=triton_url, ssl=False)

    texts = ["Hello world", "This is a test"]
    task = "retrieval.passage"
    instruction = "Represent the document for retrieval: "
    prefixed = [instruction + t for t in texts]

    encoded = tokenizer(
        prefixed, padding=True, truncation=True, max_length=768, return_tensors="np"
    )

    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)
    task_id = np.array([[TASK_IDS[task]]] * len(texts), dtype=np.int64)

    inputs = [
        grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
        grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        grpcclient.InferInput("task_id", task_id.shape, "INT64"),
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)
    inputs[2].set_data_from_numpy(task_id)

    outputs = [grpcclient.InferRequestedOutput("embedding")]

    print("Calling infer...")
    result = await client.infer(
        model_name="jina-v3-ensemble",
        inputs=inputs,
        outputs=outputs,
        headers=headers,
    )

    embeddings = result.as_numpy("embedding")
    print(f"Success! Got embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 5 dims): {embeddings[0][:5]}")


if __name__ == "__main__":
    asyncio.run(test_encode())
