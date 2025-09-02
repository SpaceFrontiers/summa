"""Harness regressions: aliases/target boundaries and failed child execution."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import check_search


class SearchHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "docs").mkdir()
        (self.root / "AGENTS.md").write_text(
            "[Contract](docs/search-system-contract.md)"
        )
        (self.root / "docs/search-system-contract.md").write_text("System rules")
        (self.root / "Cargo.toml").write_text(
            '[workspace.dependencies]\nmodel = { package = "hermes-llm", version = "1" }\n'
        )
        for name in (
            "hermes-core",
            "hermes-server",
            "hermes-broker",
            "hermes-tool",
            "hermes-wasm",
        ):
            (self.root / name).mkdir()
            (self.root / name / "Cargo.toml").write_text("[dependencies]\n")

    def manifest(self, text):
        (self.root / "hermes-core/Cargo.toml").write_text(text)

    def test_target_specific_aliased_runtime_dependency_is_rejected(self):
        self.manifest(
            "[target.'cfg(unix)'.dependencies]\nmodel = { workspace = true }\n"
        )
        with self.assertRaisesRegex(ValueError, "hermes-llm"):
            check_search.contracts(self.root)

    def test_transport_stays_in_adapters_but_dev_fixtures_can_cross_boundaries(self):
        self.manifest("[dev-dependencies]\nmodel = { workspace = true }\n")
        check_search.contracts(self.root)
        self.manifest(
            '[dependencies]\ntransport = { package = "tonic", version = "1" }\n'
        )
        with self.assertRaisesRegex(ValueError, "transport/CLI"):
            check_search.contracts(self.root)

    def test_missing_contract_link_is_reported(self):
        (self.root / "docs/search-system-contract.md").write_text(
            "[Missing](missing.md)"
        )
        with self.assertRaisesRegex(ValueError, "missing.md"):
            check_search.contracts(self.root)

    def test_child_failure_preserves_exit_status_and_output(self):
        log = self.root / "failure.log"
        code = check_search.run_command(
            [sys.executable, "-c", "print('regression evidence'); raise SystemExit(7)"],
            log,
            5,
            os.environ.copy(),
        )
        self.assertEqual(code, 7)
        self.assertIn("regression evidence", log.read_text())

    def test_timeout_terminates_child(self):
        log = self.root / "timeout.log"
        with self.assertRaises(subprocess.TimeoutExpired):
            check_search.run_command(
                [
                    sys.executable,
                    "-c",
                    "import os, time; print(os.getpid(), flush=True); time.sleep(30)",
                ],
                log,
                0.3,
                os.environ.copy(),
            )
        with self.assertRaises(ProcessLookupError):
            os.kill(int(log.read_text().strip()), 0)


if __name__ == "__main__":
    unittest.main()
