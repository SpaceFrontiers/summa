"""Publishing consumes reviewed versions and cannot change protected main."""

import json
import tempfile
import unittest
from pathlib import Path

from publish_version import publish, release_version

HEAD = "a" * 40


class Git:
    def __init__(self, existing="", dirty=False):
        self.commands = []
        self.existing = existing
        self.dirty = dirty

    def run(self, *args):
        self.commands.append(args)
        if args[:2] == ("git", "push") and args != (
            "git",
            "push",
            "origin",
            "refs/tags/v2.0.1",
        ):
            raise AssertionError("GH006: direct branch updates require a PR")
        if args == ("git", "status", "--porcelain"):
            return " M Cargo.toml" if self.dirty else ""
        if args == ("git", "rev-parse", "HEAD"):
            return HEAD
        if args[:2] == ("git", "ls-remote"):
            return self.existing
        if args[:2] == ("git", "rev-parse"):
            return self.existing
        return ""


class PublishVersionTests(unittest.TestCase):
    def test_protected_main_never_receives_a_direct_push(self):
        git = Git()
        self.assertEqual(publish("2.0.1", git.run), HEAD)
        self.assertIn(("git", "tag", "v2.0.1", HEAD), git.commands)
        self.assertIn(("git", "push", "origin", "refs/tags/v2.0.1"), git.commands)

    def test_retry_reuses_tag_at_the_same_commit(self):
        git = Git(existing=HEAD)
        self.assertEqual(publish("2.0.1", git.run), HEAD)
        self.assertFalse(any(c[:2] == ("git", "push") for c in git.commands))

    def test_existing_tag_cannot_be_moved_to_another_commit(self):
        git = Git(existing="b" * 40)
        with self.assertRaisesRegex(ValueError, "another commit"):
            publish("2.0.1", git.run)
        self.assertFalse(any(c[:2] == ("git", "push") for c in git.commands))

    def test_uncommitted_release_changes_cannot_be_published(self):
        git = Git(dirty=True)
        with self.assertRaisesRegex(ValueError, "clean, committed"):
            publish("2.0.1", git.run)
        self.assertFalse(any(c[:2] == ("git", "tag") for c in git.commands))

    def test_release_requires_matching_manifests_and_lockfiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "Cargo.toml": '[workspace.package]\nversion = "2.0.1"\n[workspace.dependencies]\nsumma-core = {path = "summa-core", version = "2.0.1"}\n',
                "Cargo.lock": '[[package]]\nname = "summa-core"\nversion = "2.0.1"\n',
                "summa-client-python/pyproject.toml": '[project]\nversion = "2.0.1"\n',
                "summa-mal-python/pyproject.toml": '[project]\nversion = "2.0.1"\n',
                "summa-client-typescript/package.json": json.dumps(
                    {"version": "2.0.1"}
                ),
                "summa-client-python/uv.lock": '[[package]]\nname = "summa-client-python"\nversion = "2.0.1"\n',
            }
            for name, text in files.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(text)
            self.assertEqual(release_version(root, "2.0.1"), "2.0.1")
            with self.assertRaisesRegex(ValueError, "Requested"):
                release_version(root, "2.0.2")
            for name, text in files.items():
                if name == "Cargo.toml":
                    continue
                with self.subTest(file=name):
                    (root / name).write_text(text.replace("2.0.1", "2.0.0"))
                    with self.assertRaisesRegex(ValueError, "must all match"):
                        release_version(root)
                    (root / name).write_text(text)


if __name__ == "__main__":
    unittest.main()
