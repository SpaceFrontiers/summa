"""Release tags must follow a protected-branch PR merge, never a direct push."""

import json
import subprocess
import unittest

from publish_version import publish

HEAD = "a" * 40
MERGE = "b" * 40


class GitHub:
    def __init__(self, state="MERGED", merge_error=False):
        self.commands = []
        self.state = state
        self.merge_error = merge_error
        self.head = HEAD
        self.changed = "Cargo.toml\nCargo.lock"

    def run(self, *args, **kwargs):
        self.commands.append(args)
        if args == ("git", "diff", "--name-only", "HEAD"):
            return self.changed
        if args[:2] == ("git", "push") and (
            len(args) == 2 or any(arg.endswith(":refs/heads/main") for arg in args)
        ):
            raise subprocess.CalledProcessError(1, args, stderr="GH006: PR required")
        if args[:3] == ("git", "rev-parse", "HEAD"):
            return HEAD
        if args[:3] == ("gh", "pr", "create"):
            return "https://github.com/test/repo/pull/1"
        if args[:3] == ("gh", "pr", "merge") and self.merge_error:
            raise subprocess.CalledProcessError(1, args, stderr="Review required")
        if args[:3] == ("gh", "pr", "view"):
            return json.dumps(
                {
                    "state": self.state,
                    "headRefOid": self.head,
                    "mergeCommit": {"oid": MERGE},
                }
            )
        return ""


class PublishVersionTests(unittest.TestCase):
    def publish(self, github):
        return publish("2.0.1", "release/v2.0.1-123-1", "main", "test/repo", github.run)

    def test_protected_main_is_updated_only_through_pr_and_tag_uses_merge_commit(self):
        github = GitHub()
        self.assertEqual(self.publish(github), MERGE)
        self.assertIn(
            ("git", "push", "origin", "HEAD:refs/heads/release/v2.0.1-123-1"),
            github.commands,
        )
        merge = next(c for c in github.commands if c[:3] == ("gh", "pr", "merge"))
        self.assertIn("--match-head-commit", merge)
        self.assertIn(HEAD, merge)
        self.assertNotIn("--admin", merge)
        self.assertIn(("git", "tag", "v2.0.1", MERGE), github.commands)
        self.assertLess(
            github.commands.index(merge),
            github.commands.index(("git", "tag", "v2.0.1", MERGE)),
        )

    def test_required_review_blocks_tagging_and_publication(self):
        github = GitHub(merge_error=True)
        with self.assertRaises(subprocess.CalledProcessError):
            self.publish(github)
        self.assertFalse(any(c[:2] == ("git", "tag") for c in github.commands))

    def test_queued_merge_does_not_create_a_release_tag(self):
        github = GitHub(state="OPEN")
        with self.assertRaisesRegex(ValueError, "not merged"):
            self.publish(github)
        self.assertFalse(any(c[:2] == ("git", "tag") for c in github.commands))

    def test_changed_pr_head_cannot_be_published(self):
        github = GitHub()
        github.head = "c" * 40
        with self.assertRaisesRegex(ValueError, "head changed"):
            self.publish(github)
        self.assertFalse(any(c[:2] == ("git", "tag") for c in github.commands))

    def test_unrelated_changes_are_not_pushed_with_release(self):
        github = GitHub()
        github.changed = "Cargo.toml\nsumma-core/src/lib.rs"
        with self.assertRaisesRegex(ValueError, "Unexpected release changes"):
            self.publish(github)
        self.assertFalse(any(c[:2] == ("git", "push") for c in github.commands))

    def test_release_files_changed_during_merge_prevent_tagging(self):
        github = GitHub()
        run = github.run

        def changed_version(*args, **kwargs):
            if args[:3] == ("git", "diff", "--exit-code"):
                raise subprocess.CalledProcessError(1, args)
            return run(*args, **kwargs)

        with self.assertRaises(subprocess.CalledProcessError):
            publish(
                "2.0.1", "release/v2.0.1-123-1", "main", "test/repo", changed_version
            )
        self.assertFalse(any(c[:2] == ("git", "tag") for c in github.commands))


if __name__ == "__main__":
    unittest.main()
