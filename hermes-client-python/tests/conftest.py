"""Shared fixtures for integration tests.

Starts hermes-server in a subprocess and provides a connected HermesClient.
"""

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import pytest_asyncio
from hermes_client_python.client import HermesClient

SERVER_BINARY = (
    Path(__file__).resolve().parents[2] / "target" / "debug" / "hermes-server"
)
SERVER_PORT = 50052  # Avoid clashing with a running dev server on 50051


@pytest.fixture(scope="session")
def server():
    """Start hermes-server for the test session."""
    data_dir = tempfile.mkdtemp(prefix="hermes_test_")
    proc = subprocess.Popen(
        [
            str(SERVER_BINARY),
            "--data-dir",
            data_dir,
            "--addr",
            f"0.0.0.0:{SERVER_PORT}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for server to be ready
    time.sleep(2)
    assert proc.poll() is None, f"Server failed to start: {proc.stderr.read().decode()}"
    yield proc
    proc.terminate()
    proc.wait(timeout=5)
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest_asyncio.fixture
async def client(server):
    """Provide a connected HermesClient."""
    async with HermesClient(f"localhost:{SERVER_PORT}") as c:
        yield c
