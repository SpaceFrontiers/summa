"""Regression coverage for Markdown parsing and navigation checks."""

import tempfile
import unittest
from pathlib import Path

from check_docs import check, parse_document


class DocumentationChecks(unittest.TestCase):
    def test_reference_links_fences_unicode_and_duplicate_headings(self):
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "guide.md"
            page.write_text(
                "# Hello `API`\n\n# Hello API\n\n# Привет мир\n\n"
                "[reference][target]\n\n[target]: other.md#hello\n\n"
                "[balanced](other(file).md)\n\n![image](plot.svg)\n\n"
                "<a id='custom'></a>\n\n"
                "```md\n# Fake heading\n[example](missing.md)\n```\n"
            )
            anchors, links = parse_document(page)
            self.assertEqual(
                anchors, {"hello-api", "hello-api-1", "привет-мир", "custom"}
            )
            self.assertEqual(
                [url for url, _ in links],
                ["other.md#hello", "other(file).md", "plot.svg"],
            )

    def test_missing_links_anchors_navigation_and_benchmark_targets_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "docs").mkdir()
            (root / "summa-core").mkdir()
            (root / "Cargo.toml").write_text('[workspace]\nmembers = ["summa-core"]\n')
            (root / "summa-core/Cargo.toml").write_text(
                '[package]\nname = "summa-core"\n[[bench]]\nname = "real"\n'
            )
            (root / "docs/README.md").write_text(
                "# Guides\n\n[bench](benchmarks.md)\n\n"
                "[missing](missing.md)\n\n[anchor](benchmarks.md#missing)\n"
            )
            (root / "docs/benchmarks.md").write_text(
                "# Benchmarks\n\n| `summa-core` | `removed` |\n"
            )
            (root / "docs/orphan.md").write_text("# Orphan\n")
            errors, _, _ = check(root, list(root.rglob("*.md")))
            report = "\n".join(errors)
            for message in [
                "missing file: missing.md",
                "missing heading: benchmarks.md#missing",
                "guide is not linked: orphan.md",
                "missing target: summa-core/real",
                "unknown target: summa-core/removed",
            ]:
                self.assertIn(message, report)


if __name__ == "__main__":
    unittest.main()
