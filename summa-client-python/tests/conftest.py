"""Shared fixtures for integration tests.

Starts summa-server in a subprocess and provides a connected SummaClient.
"""

import subprocess
import tempfile
import time
from pathlib import Path

import grpc
import pytest
import pytest_asyncio
from summa_client_python.client import SummaClient

SERVER_BINARY = (
    Path(__file__).resolve().parents[2] / "target" / "debug" / "summa-server"
)
SERVER_PORT = 50052  # Avoid clashing with a running dev server on 50051
SERVER_ADDRESS = f"127.0.0.1:{SERVER_PORT}"
SERVER_STARTUP_TIMEOUT = 60


def _wait_for_server(proc, log):
    """Wait for a gRPC handshake, failing promptly if the child exits."""
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    with grpc.insecure_channel(SERVER_ADDRESS) as channel:
        ready = grpc.channel_ready_future(channel)
        try:
            while proc.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    ready.result(timeout=min(0.1, remaining))
                    return
                except grpc.FutureTimeoutError:
                    pass
        finally:
            ready.cancel()
    log.seek(0, 2)
    log.seek(max(0, log.tell() - 65536))
    output = log.read(65536).decode(errors="replace")
    reason = (
        f"exited with status {proc.returncode}"
        if proc.returncode is not None
        else f"did not become ready at {SERVER_ADDRESS} within {SERVER_STARTUP_TIMEOUT}s"
    )
    pytest.fail(f"summa-server {reason}\nLast server output:\n{output}")


@pytest.fixture(scope="session")
def server():
    """Start summa-server and release its process/files even if setup fails."""
    # Files avoid filling an undrained subprocess pipe during long tests.
    with (
        tempfile.TemporaryDirectory(prefix="summa_test_") as data_dir,
        tempfile.TemporaryFile() as log,
    ):
        proc = subprocess.Popen(
            [
                str(SERVER_BINARY),
                "--data-dir",
                data_dir,
                "--addr",
                SERVER_ADDRESS,
            ],
            stdout=log,
            stderr=log,
        )
        try:
            _wait_for_server(proc, log)
            yield proc
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


@pytest_asyncio.fixture
async def client(server):
    """Provide a connected SummaClient."""
    async with SummaClient(SERVER_ADDRESS) as c:
        yield c
