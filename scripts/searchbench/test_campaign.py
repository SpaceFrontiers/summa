"""Regression coverage for the observed-result comparison gate."""

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import campaign


@dataclass
class Item:
    query_class: str
    text: str


class AgreementGateTests(unittest.TestCase):
    def test_regex_is_included_only_when_every_engine_reports_the_same_count(self):
        items = [
            Item("regex", "[0-9]{4}"),
            Item("regex", ".*0th"),
            Item("high_term", "file"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            for engine in campaign.ENGINES:
                rows = [
                    {"key": campaign.key(items[0]), "count": 42},
                    {"key": campaign.key(items[1]), "count": 7},
                    {"key": campaign.key(items[2]), "count": 100},
                ]
                if engine == "summa":
                    rows[1] = {
                        "key": campaign.key(items[1]),
                        "error": "expansion budget exceeded",
                    }
                    rows[2]["count"] = 101
                (out / f"{engine}-counts.json").write_text(json.dumps(rows))
            campaign.gate(SimpleNamespace(out=out), items)
            result = json.loads((out / "agreement.json").read_text())
            self.assertEqual([row["include"] for row in result], [True, False, False])
            self.assertEqual(
                result[1]["observed"]["summa"]["error"], "expansion budget exceeded"
            )

    def test_shared_input_gate_retains_count_differences_and_excludes_errors(self):
        item = Item("high_term", "file")
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            for engine in campaign.ENGINES:
                (out / f"{engine}-counts.json").write_text(
                    json.dumps(
                        [
                            {
                                "key": campaign.key(item),
                                "count": 101 if engine == "summa" else 100,
                            }
                        ]
                    )
                )
            args = SimpleNamespace(out=out, comparison="shared-input")
            campaign.gate(args, [item])
            result = json.loads((out / "agreement.json").read_text())
            self.assertTrue(result[0]["include"])
            self.assertFalse(result[0]["counts_agree"])
            self.assertEqual(result[0]["comparison"], "shared-input")
            self.assertEqual(
                campaign.eligible_counts(result, "summa"), {campaign.key(item): 101}
            )
            self.assertEqual(
                campaign.eligible_counts(result, "elasticsearch"),
                {campaign.key(item): 100},
            )
            (out / "summa-counts.json").write_text(
                json.dumps([{"key": campaign.key(item), "error": "budget exceeded"}])
            )
            campaign.gate(args, [item])
            self.assertFalse(
                json.loads((out / "agreement.json").read_text())[0]["include"]
            )


if __name__ == "__main__":
    unittest.main()
